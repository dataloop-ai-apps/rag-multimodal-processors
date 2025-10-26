"""
PDF Processor Application for Dataloop.
Extracts text from PDFs, applies OCR, and creates chunks for RAG.
"""

from typing import List, Dict, Any
import dtlpy as dl
import logging
import tempfile
import fitz
import pymupdf4llm
import os
import sys

# Add parent directories to path for shared imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from chunkers.text_chunker import TextChunker
from extractors.ocr_extractor import OCRExtractor
from utils.text_cleaning import clean_text
from utils.dataloop_helpers import upload_chunks, cleanup_temp_items_and_folder
from utils.chunk_metadata import ChunkMetadata

logger = logging.getLogger('pdf-processor')


class PDFProcessor(dl.BaseServiceRunner):
    """
    PDF Processor for extracting text, applying OCR, and creating chunks.
    
    Supports:
    - Text extraction (plain and markdown-aware)
    - Image extraction and OCR
    - Multiple chunking strategies
    - Text cleaning and normalization
    """

    def __init__(self):
        """Initialize the PDF processor."""
        # Configure Dataloop client timeouts
        dl.client_api._upload_session_timeout = 60
        dl.client_api._upload_chuck_timeout = 30
        
        # Download required NLTK data (only if not already present)
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            logger.info("Downloading NLTK averaged_perceptron_tagger")
            nltk.download('averaged_perceptron_tagger', quiet=True)
        
        logger.info("PDFProcessor initialized")

    def process_document(self, item: dl.Item, target_dataset: dl.Dataset, context: dl.Context) -> List[dl.Item]:
        """
        Main entry point for PDF processing.
        
        Args:
            item (dl.Item): PDF item to process
            target_dataset (dl.Dataset): Target dataset for storing chunks
            context (dl.Context): Processing context with configuration
            
        Returns:
            List[dl.Item]: List of chunk items
        """
        logger.info(
            f"Processing PDF | item_id={item.id} name={item.name} mimetype={item.mimetype} "
            f"target_dataset={target_dataset.name}"
        )
        
        # Get configuration from node
        node = context.node
        config = node.metadata['customNodeConfig']
        
        # Extract configuration parameters
        ocr_from_images = config.get('ocr_from_images', False)
        ocr_integration_method = config.get('ocr_integration_method', 'append_to_page')
        use_markdown_extraction = config.get('use_markdown_extraction', False)
        chunking_strategy = config.get('chunking_strategy', 'recursive')
        max_chunk_size = config.get('max_chunk_size', 300)
        chunk_overlap = config.get('chunk_overlap', 20)
        to_correct_spelling = config.get('to_correct_spelling', False)
        
        # Hardcoded values (not configurable by user)
        remote_path_for_chunks = '/chunks'

        logger.info(
            f"Config | markdown={use_markdown_extraction} ocr_from_images={ocr_from_images} "
            f"ocr_method=EasyOCR strategy={chunking_strategy} chunk_size={max_chunk_size}"
        )

        # Extract content
        combined_text, total_pages = self._extract_content(
            item, ocr_from_images, ocr_integration_method, use_markdown_extraction
        )
        
        # Create chunker
        chunker = TextChunker(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunking_strategy,
            use_markdown_splitting=use_markdown_extraction
        )
        
        # Create chunks
        chunks = chunker.chunk(combined_text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Apply cleaning if requested
        if to_correct_spelling:
            logger.info("Applying text cleaning")
            chunks = [clean_text(chunk) for chunk in chunks]
        
        # Get processor-specific metadata
        processor_metadata = self._get_metadata(total_pages, use_markdown_extraction)
        processor_metadata['chunking_strategy'] = chunking_strategy
        
        # Upload chunks
        chunked_items = upload_chunks(
            chunks=chunks,
            original_item=item,
            target_dataset=target_dataset,
            remote_path=remote_path_for_chunks,
            processor_metadata=processor_metadata
        )
        
        logger.info(
            f"Processing completed | chunks={len(chunked_items)} dataset={target_dataset.name}"
        )
        return chunked_items

    def _extract_content(self, item: dl.Item, ocr_from_images: bool,
                        ocr_integration_method: str, use_markdown_extraction: bool) -> tuple[str, int]:
        """
        Extract content from PDF item.
        
        Returns:
            tuple: (combined_text, total_pages)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            item_local_path = item.download(local_path=temp_dir)
            logger.info(f"Downloaded item | path={item_local_path}")
            
            if ocr_from_images and not use_markdown_extraction:
                page_texts, total_pages, images = self._extract_text_and_images(item_local_path, item.id)
                
                logger.info(f"OCR enabled | using EasyOCR")
                ocr_extractor = OCRExtractor(model_id=None)
                ocr_texts = self._process_images_with_easyocr(images, item.id, ocr_extractor)
            else:
                # Extract text and images separately (markdown or no OCR not used)
                if use_markdown_extraction:
                    page_texts, total_pages = self._extract_text_as_markdown(item_local_path, item.id)
                else:
                    page_texts, total_pages = self._extract_text(item_local_path, item.id)
                
                ocr_texts = []
                if ocr_from_images:
                    logger.info(f"OCR enabled | using EasyOCR")
                    ocr_extractor = OCRExtractor(model_id=None)
                    ocr_texts = self._extract_and_ocr_with_easyocr(
                        item_local_path, item.id, ocr_extractor
                    )
            
            combined_text = self._combine_texts(page_texts, ocr_texts, ocr_integration_method)
            logger.info(f"Combined text | length={len(combined_text)} pages={total_pages}")
            
            return combined_text, total_pages

    def _extract_text(self, pdf_path: str, item_id: str) -> tuple[List[str], int]:
        """
        Extract text from PDF using PyMuPDF.
        
        Returns:
            tuple: (page_texts, total_pages)
        """
        logger.info(f"Extracting text | item_id={item_id}")
        
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            page_texts = []
            for page in doc:
                page_texts.append(page.get_text())
            
            logger.info(f"Extracted {len(page_texts)} pages")
            return page_texts, total_pages
    
    def _extract_text_and_images(self, pdf_path: str, item_id: str) -> tuple[List[str], int, List[Dict[str, Any]]]:
        """
        Extract both text and images from PDF in a single pass.
        
        Returns:
            tuple: (page_texts, total_pages, images)
        """
        logger.info(f"Extracting text and images in single pass | item_id={item_id}")
        
        page_texts = []
        images = []
        
        with fitz.open(pdf_path) as pdf_file:
            total_pages = len(pdf_file)
            
            for page_index in range(total_pages):
                page = pdf_file.load_page(page_index)
                
                # Extract text from this page
                page_texts.append(page.get_text())
                
                # Extract images from this page
                image_list = page.get_images(full=True)
                for image_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_file.extract_image(xref)
                        images.append({
                            'bytes': base_image["image"],
                            'page_index': page_index,
                            'image_index': image_index,
                            'extension': base_image.get("ext", "png")
                        })
                    except Exception as e:
                        logger.warning(f"Failed to extract image page={page_index} img={image_index}: {e}")
                        continue
        
        logger.info(f"Extracted {total_pages} pages and {len(images)} images in single pass")
        return page_texts, total_pages, images

    def _extract_text_as_markdown(self, pdf_path: str, item_id: str) -> tuple[List[str], int]:
        """
        Extract text as markdown using pymupdf4llm.
        
        Returns:
            tuple: (page_texts, total_pages)
        """
        logger.info(f"Extracting markdown | item_id={item_id}")
        
        try:
            md_text = pymupdf4llm.to_markdown(
                pdf_path,
                page_chunks=True,
                write_images=False,
                show_progress=True
            )
            
            if isinstance(md_text, list):
                page_texts = []
                for page_data in md_text:
                    if isinstance(page_data, dict):
                        page_texts.append(page_data.get('text', ''))
                    else:
                        page_texts.append(str(page_data))
            elif isinstance(md_text, str):
                page_texts = md_text.split('\n-----\n')
            else:
                logger.warning("Unexpected markdown format, falling back")
                return self._extract_text(pdf_path, item_id)
            
            total_pages = len(page_texts)
            logger.info(f"Extracted {total_pages} pages as markdown")
            return page_texts, total_pages
            
        except Exception as e:
            logger.exception(f"Markdown extraction failed: {e}")
            return self._extract_text(pdf_path, item_id)

    def _extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract all images from PDF.
        
        Returns:
            List of dicts with: bytes, page_index, image_index, extension
        """
        images = []
        
        with fitz.open(pdf_path) as pdf_file:
            for page_index in range(len(pdf_file)):
                page = pdf_file.load_page(page_index)
                image_list = page.get_images(full=True)
                
                for image_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_file.extract_image(xref)
                        images.append({
                            'bytes': base_image["image"],
                            'page_index': page_index,
                            'image_index': image_index,
                            'extension': base_image.get("ext", "png")
                        })
                    except Exception as e:
                        logger.warning(f"Failed to extract image page={page_index} img={image_index}: {e}")
                        continue
        
        logger.info(f"Extracted {len(images)} images from PDF")
        return images
    
    def _process_images_with_easyocr(self, images: List[Dict[str, Any]], item_id: str,
                                     ocr_extractor: OCRExtractor) -> List[Dict[str, Any]]:
        """
        Process already-extracted images with EasyOCR.
        Uses temporary files, no upload to Dataloop.
        
        Args:
            images: List of image dictionaries with 'bytes', 'page_index', 'image_index', 'extension'
            item_id: Item ID for logging
            ocr_extractor: OCR extractor instance
            
        Returns:
            List of OCR results with page/image indices
        """
        if not images:
            logger.info("No images to process")
            return []
        
        logger.info(f"Processing {len(images)} images with EasyOCR | item_id={item_id}")
        
        ocr_results = []
        for image_data in images:
            try:
                # Save to temp file for EasyOCR processing
                temp_image_path = os.path.join(
                    tempfile.gettempdir(),
                    f"ocr_img_{item_id}_{image_data['page_index']}_{image_data['image_index']}.{image_data['extension']}"
                )
                
                with open(temp_image_path, 'wb') as f:
                    f.write(image_data['bytes'])
                
                # Run EasyOCR on local file
                ocr_text = ocr_extractor.extract_text(temp_image_path)
                
                # Clean up temp file
                try:
                    os.remove(temp_image_path)
                except:
                    pass
                
                ocr_results.append({
                    'page_index': image_data['page_index'],
                    'image_index': image_data['image_index'],
                    'text': ocr_text,
                    'extension': image_data['extension']
                })
            except Exception as e:
                logger.warning(f"EasyOCR failed on image page={image_data['page_index']} img={image_data['image_index']}: {e}")
                continue
        
        logger.info(f"EasyOCR completed | images_processed={len(ocr_results)}")
        return ocr_results
    
    def _extract_and_ocr_with_easyocr(self, pdf_path: str, item_id: str,
                                      ocr_extractor: OCRExtractor) -> List[Dict[str, Any]]:
        """
        Extract images from PDF and apply OCR using EasyOCR (local processing).
        Uses temporary files, no upload to Dataloop.
        """
        logger.info(f"Extracting images for EasyOCR | item_id={item_id}")
        
        images = self._extract_images_from_pdf(pdf_path)
        return self._process_images_with_easyocr(images, item_id, ocr_extractor)
    
    def _extract_and_ocr_with_dataloop_model_from_images(self, images: List[Dict[str, Any]], 
                                                         original_item: dl.Item,
                                                         ocr_extractor: OCRExtractor) -> List[Dict[str, Any]]:
        """
        Process already-extracted images with custom Dataloop OCR model.
        Flow: Upload → Predict → Get text → Delete items + folder
        
        Args:
            images: List of image dictionaries with 'bytes', 'page_index', 'image_index', 'extension'
            original_item: Original PDF item
            ocr_extractor: OCR extractor instance
            
        Returns:
            List of OCR results with page/image indices
        """
        if not images:
            logger.info("No images to process")
            return []
        
        logger.info(f"Processing {len(images)} images with Dataloop OCR model | item_id={original_item.id}")
        
        # Setup temp directories
        temp_folder_name = f"./dataloop/temp_images_ocr_{original_item.name}"
        temp_local_dir = tempfile.mkdtemp(prefix=f"ocr_images_{original_item.id}_")
        
        try:
            # Save images locally and upload to Dataloop
            uploaded_items = self._upload_images_to_dataloop(
                images, temp_local_dir, temp_folder_name, original_item.dataset
            )
            
            # Run batch OCR
            ocr_results = self._run_batch_ocr(uploaded_items, ocr_extractor)
            
            return ocr_results
            
        finally:
            # Cleanup all temporary resources
            uploaded_item_objects = [item for item, _ in uploaded_items] if uploaded_items else []
            cleanup_temp_items_and_folder(
                uploaded_item_objects,
                temp_folder_name,
                original_item.dataset,
                temp_local_dir
            )
    
    def _extract_and_ocr_with_dataloop_model(self, pdf_path: str, original_item: dl.Item,
                                             ocr_extractor: OCRExtractor) -> List[Dict[str, Any]]:
        """
        Extract images from PDF, upload to Dataloop, run custom OCR model, then cleanup.
        Flow: Extract → Upload → Predict → Get text → Delete items + folder
        """
        logger.info(f"Extracting images for Dataloop OCR model | item_id={original_item.id}")
        
        # Extract images from PDF
        images = self._extract_images_from_pdf(pdf_path)
        return self._extract_and_ocr_with_dataloop_model_from_images(images, original_item, ocr_extractor)
    
    def _upload_images_to_dataloop(
        self,
        images: List[Dict[str, Any]],
        local_dir: str,
        remote_folder: str,
        dataset: dl.Dataset
     ) -> List[tuple]:
        """
        Save images locally and upload to Dataloop in batch.
        
        Returns:
            List of tuples: (uploaded_item, metadata)
        """
        if not images:
            return []
        
        # Step 1: Save all images locally
        image_paths = []
        image_metadata_list = []
        
        for image_data in images:
            try:
                # Save to local temp directory
                image_filename = f"page_{image_data['page_index']}_img_{image_data['image_index']}.{image_data['extension']}"
                image_path = os.path.join(local_dir, image_filename)
                
                with open(image_path, 'wb') as f:
                    f.write(image_data['bytes'])
                
                image_paths.append(image_path)
                image_metadata_list.append({
                    'page_index': image_data['page_index'],
                    'image_index': image_data['image_index'],
                    'extension': image_data['extension']
                })
                
            except Exception as e:
                logger.warning(f"Failed to save image page={image_data['page_index']}: {e}")
                continue
        
        if not image_paths:
            logger.warning("No images could be saved locally")
            return []
        
        # Step 2: Batch upload to Dataloop
        logger.info(f"Batch uploading {len(image_paths)} images to Dataloop | folder={remote_folder}")
        try:
            uploaded = dataset.items.upload(
                local_path=image_paths,
                remote_path=remote_folder,
                overwrite=True
            )
            
            # Handle single item vs list response
            if isinstance(uploaded, dl.Item):
                uploaded_items_list = [uploaded]
            else:
                uploaded_items_list = list(uploaded)
            
            # Pair uploaded items with their metadata
            uploaded_items = []
            for idx, uploaded_item in enumerate(uploaded_items_list):
                if idx < len(image_metadata_list):
                    uploaded_items.append((uploaded_item, image_metadata_list[idx]))
                    logger.debug(f"Uploaded image | item_id={uploaded_item.id} page={image_metadata_list[idx]['page_index']}")
            
            logger.info(f"Batch upload completed | uploaded={len(uploaded_items)} images")
            return uploaded_items
            
        except Exception as e:
            logger.error(f"Batch upload failed: {e}, falling back to individual uploads")
            uploaded_items = []
            for image_path, metadata in zip(image_paths, image_metadata_list):
                try:
                    uploaded_item = dataset.items.upload(
                        local_path=image_path,
                        remote_path=remote_folder,
                        overwrite=True
                    )
                    uploaded_items.append((uploaded_item, metadata))
                except Exception as upload_err:
                    logger.warning(f"Failed to upload {image_path}: {upload_err}")
                    continue
            
            logger.info(f"Individual uploads completed | uploaded={len(uploaded_items)} images")
            return uploaded_items
    
    def _run_batch_ocr(
        self,
        uploaded_items: List[tuple],
        ocr_extractor: OCRExtractor
     ) -> List[Dict[str, Any]]:
        """
        Run batch OCR on uploaded items.
        
        Args:
            uploaded_items: List of (item, metadata) tuples
            ocr_extractor: OCR extractor instance
            
        Returns:
            List of OCR results with page/image indices
        """
        uploaded_item_objects = [item for item, _ in uploaded_items]
        ocr_text_results = ocr_extractor.extract_text_batch(uploaded_item_objects)
        
        # Map results back to original metadata
        ocr_results = []
        for uploaded_item, metadata in uploaded_items:
            ocr_text = ocr_text_results.get(uploaded_item.id, "")
            ocr_results.append({
                'page_index': metadata['page_index'],
                'image_index': metadata['image_index'],
                'text': ocr_text,
                'extension': metadata['extension']
            })
        
            logger.info(f"OCR completed | successful={sum(1 for r in ocr_results if r['text'])}/{len(uploaded_items)}")
        return ocr_results

    def _combine_texts(self, page_texts: List[str], ocr_texts: List[Dict[str, Any]],
                      integration_method: str) -> str:
        """Combine PDF text and OCR text."""
        if integration_method == 'append_to_page':
            combined_pages = page_texts.copy()
            for ocr_result in ocr_texts:
                page_idx = ocr_result['page_index']
                if page_idx < len(combined_pages):
                    combined_pages[page_idx] += f"\n\n[OCR_IMAGE_{ocr_result['image_index']}]\n{ocr_result['text']}" if ocr_result['text'] else ''
            return '\n\n'.join(combined_pages)
            
        elif integration_method == 'separate_chunks':
            pdf_text = '\n\n'.join(page_texts)
            ocr_text = '\n\n'.join([f"[OCR_PAGE_{r['page_index']}_IMAGE_{r['image_index']}]\n{r['text']}" if r['text'] else ''
                                   for r in ocr_texts])
            return f"{pdf_text}\n\n[OCR_SECTION]\n{ocr_text}"
            
        else:  # combine_all
            all_text = '\n\n'.join(page_texts)
            for ocr_result in ocr_texts:
                all_text += f"\n\n[OCR_PAGE_{ocr_result['page_index']}_IMAGE_{ocr_result['image_index']}]\n{ocr_result['text']}" if ocr_result['text'] else ''
            return all_text

    def _get_metadata(self, total_pages: int, use_markdown_extraction: bool) -> Dict[str, Any]:
        """
        Get processor-specific metadata.
        
        Args:
            total_pages: Total number of pages in the PDF
            use_markdown_extraction: Whether markdown extraction was used
            
        Returns:
            Dict with processor-specific metadata fields
        """
        return {
            'total_pages': total_pages,
            'extraction_method': 'pymupdf4llm' if use_markdown_extraction else 'fitz',
            'extraction_format': 'markdown' if use_markdown_extraction else 'plain',
            'markdown_aware_splitting': use_markdown_extraction,
        }

