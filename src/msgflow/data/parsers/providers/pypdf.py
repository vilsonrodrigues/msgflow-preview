from pypdf import PdfReader
import os

from msgflow.data.parsers.base import BaseParser
from msgflow.data.parsers.types import PdfParser

# TODO: convert image ot base64
class PyPDFPdfParser(BaseParser, PdfParser):

    def pdf_parser(path):
        """
        Parser de PDF que converte o conteúdo em Markdown e extrai imagens.
        
        Args:
            path (str): Caminho do arquivo PDF.
            save_imgs (str, optional): Diretório para salvar as imagens extraídas.
        
        Returns:
            tuple: (conteúdo em Markdown, dicionário com imagens {nome: dados})
        """
        def _convert(path):
            md_content = ""
            image_dict = {}
            reader = PdfReader(path)
            num_pages = len(reader.pages)
            
            # Itera sobre cada página do PDF
            for idx in range(num_pages):
                page = reader.pages[idx]
                
                # Adiciona um comentário com o número da página (similar ao PPTX)
                md_content += f"\n\n<!-- Page number: {idx + 1} -->\n"
                
                text = page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)
                md_content += text.strip() + "\n"
                
                for count, image_file_object in enumerate(page.images):

                    img_extension = image_file_object.name.split('.')[-1]
                    img_name = f"image_page{idx + 1}_{count}.{img_extension}"
                    img_data = image_file_object.data
                    
                    image_dict[img_name] = img_data
                    
                    md_content += f"\n![{img_name}]({img_name})\n"
                            
            return md_content.strip(), image_dict
        
        return _convert(path)
