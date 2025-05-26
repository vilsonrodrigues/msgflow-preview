# A collection of collaborative modules
from typing import Any, Dict, List, Optional, Union
from msgflow.generation.templates import AVAILABLE_MEMBERS_TEMPLATE
from msgflow.message import Message
from msgflow.models.gateway import ModelGateway
from msgflow.models.types import ChatCompletionModel
from msgflow.nn import functional as F
from msgflow.nn.modules.agent import Agent
from msgflow.nn.modules.container import ModuleDict
from msgflow.nn.modules.module import Module
from msgflow.utils.chat import ChatML, format_available_members, format_member_responses


class Collaborative(Module):

    def _set_team(self, team: List[Module]):
        if team:
            if not all(isinstance(member, Module) for member in team):
                raise ValueError("All team members must inherit from `nn.Module`")
            if not all(hasattr(member, "description") for member in team):
                raise ValueError("All team members must have a description")
            if not all(hasattr(member, "name") for member in team):
                raise ValueError("All team members must have a name")
                
            self.team = ModuleDict()
            for member in team:
                self._add_member(member)
        else:
            raise ValueError("`team` requires be a List[Modules]")

    def _add_member(self, member: Module):
        if member.name in self.team.keys():
            return # Not raise
        self.team.update({member.name: member})

    def _remove_member(self, member_name: str):
        if member_name in self.team.keys():
            self.library.pop(member_name)

    def _set_available_members_template(self, available_members_template: str):
        if isinstance(available_members_template, str):
            self.register_buffer("available_members_template", available_members_template)
        else:
            raise TypeError(f"`available_members_template` need be a str , given `{type(available_members_template)}`")

    def _set_max_iterations(self, max_iterations: int):
        if isinstance(max_iterations, int):
            if max_iterations < 2:
                raise ValueError(f"`max_iterations` need be greater than 1, given `{max_iterations}`")
            self.register_buffer("max_iterations", max_iterations)
        else:
            raise TypeError(f"`max_iterations` need be a int , given `{type(max_iterations)}`")

    def _set_model_to_new_members(self, model: Optional[Union[ChatCompletionModel, ModelGateway]] = None):
        if model.model_type == "chat_completion" or model is None:
            self.register_buffer("model_to_new_members", model)
        else:
            raise TypeError(f"`model_to_new_members` need be a `chat completion` model or None, given `{type(model)}`")


class Coordinator(Collaborative):

    def __init__(
        self,
        name: str,
        coordinator: Agent,
        team: List[Module],
        response_mode: Optional[str] = "plain_response",
        max_iterations: Optional[int] = 12,
        model_to_new_members: Optional[Union[ChatCompletionModel, ModelGateway]] = None,
        description: Optional[str] = None,
        available_members_template: Optional[str] = AVAILABLE_MEMBERS_TEMPLATE,
    ):
        super().__init__()
        self.set_name(name)
        self.set_description(description)
        self._set_available_members_template(available_members_template)
        self._set_max_iterations(max_iterations)
        self._set_model_to_new_members(model_to_new_members)
        self._set_team(team)
        self._set_coordinator(coordinator)
        self._set_response_mode(response_mode)

    def forward(self, message: Union[str, Message, Dict[str, str], List[Dict[str, Any]]]):
        history = ChatML()
        coordinator_messages = self.coordinator._prepare_task(message)
        coordinator_response = self.coordinator(coordinator_messages)
        history.extend_history(coordinator_messages)

        for iter in range(self.max_iterations):

            if coordinator_response.get("new_members", None):
                new_members = coordinator_response.pop("new_members") # Remove from history
                for member_params in new_members:
                    member_params["model"] = self.model_to_new_members or self.coordinator.model
                    self._add_member(Agent(**member_params))

            if coordinator_response.get("remove_members", None):
                remove_members = coordinator_response.pop("remove_members") # Remove from history
                for member_name in remove_members:
                    self._remove_member(member_name)

            if coordinator_response.get("tasks", None):
                tasks = []
                team_responses = []

                for task in coordinator_response["tasks"]:
                    member_module = self.team.get(member_name, None)
                    member_name = task["member"]
                    member_task = task["task"]

                    if member_module is None:
                        team_responses.append({
                            "member": member_name,
                            "task": member_task,
                            "result": "This member is not available in team"
                        })
                        continue

                    tasks.append({
                        "member": member_name,
                        "task": member_task,
                        "member_module": member_module,
                    })

                results = F.scatter_gather(
                    [item["task"] for item in tasks],
                    [item["member_module"] for item in tasks]
                )

                for task, result in zip(tasks, results):
                    task.pop("member_module")
                    if result is None:
                        result = "Task execution by this member failed"
                    task["result"] = result

                history.add_assist_message(str(coordinator_response)) # Json / XML aqui?

                team_responses.extend(tasks)
                xml_team_response = format_member_responses(team_responses, iter)
                history.add_assist_message(xml_team_response)

                coordinator_response = self.coordinator(history.get_messages())
            elif coordinator_response.get("final_answer", None):
                return self._prepare_response(coordinator_response.get("final_answer"), message)             

    def _set_coordinator(self, coordinator: Agent):        
        if isinstance(coordinator, Agent):
            if coordinator.stream == True:
                raise ValueError("Coordinator output cannot be in stream, set `stream=False`")
            if coordinator.response_mode != "plain_response":
                raise ValueError("Coordinator response must be `response_mode=plain_response`")
            if coordinator.response_template is not None:
                raise ValueError("Coordinator response needs to be structured, and "
                                 "so is not compatible with `response_template` which "
                                 "converts the output to a string, set `response_template=None`")                
            if coordinator.generation_schema is None and coordinator.xml_to_dict is None:
                raise ValueError("Coordinator requires structured output. This can be done via "
                                 "`generation_schema` or using `xml_to_dict=True`")
            self.coordinator = coordinator
            self._set_members_description_in_coordinator()
        else:
            raise TypeError(f"Coordinator requires be a `nn.Agent`, given `{type(coordinator)}`")

    def _set_members_description_in_coordinator(self):
        members_desc = []
        for member in self.team:
            members_desc.append({"name": member.name, "description": member.description})
        available_members = format_available_members(members_desc)
        template_inputs = {"members": available_members}
        if hasattr(self, "max_iterations"):
            template_inputs["max_iterations"] = self.max_iterations
        team_members = self._format_template(
            template_inputs,
            self.available_members_template
        )
        self.coordinator._set_team_members(team_members)


class Selector(Coordinator):

    def __init__(
        self,
        name: str,
        coordinator: Agent,
        team: List[Module],
        response_mode: Optional[str] = "plain_response",
        description: Optional[str] = None,
        available_members_template: Optional[str] = AVAILABLE_MEMBERS_TEMPLATE,
    ):
        super().__init__()
        self.set_name(name)
        self.set_description(description)
        self._set_available_members_template(available_members_template)
        self._set_team(team)
        self._set_coordinator(coordinator)
        self._set_response_mode(response_mode)

    def forward(self, message: Union[str, Message, Dict[str, str], List[Dict[str, Any]]]):
        coordinator_response = self.coordinator(message)
        new_query = coordinator_response.pop("new_query", None)
        member_response = self.team[coordinator_response["member"]](new_query or message)
        return self._prepare_response(member_response, message)
