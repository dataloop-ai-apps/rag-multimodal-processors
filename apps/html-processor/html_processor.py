"""
HTML processor for handling HTML files (.html, .htm).
"""

import os
from typing import Dict, Any, List
import dtlpy as dl
from pipeline.base.processor import BaseProcessor, ProcessorError
from pipeline.utils.logging_utils import ProcessorLogger, ErrorHandler, FileValidator


class HTMLProcessor(BaseProcessor):
    """
    Processor for HTML files (.html, .htm).
    """

    def __init__(self):
        """Initialize HTML processor."""
        super().__init__('html')
        self.logger = ProcessorLogger('html')
        self.error_handler = ErrorHandler('html')
        self.validator = FileValidator()

    def _extract_content(self, item: dl.Item, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from HTML file.

        Args:
            item: HTML file item
            config: Processing configuration

        Returns:
            Dictionary containing extracted content and metadata
        """
        try:
            # Download file to temporary location
            import tempfile

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = item.download(local_path=temp_dir)

                # Validate file
                if not self.validator.validate_file_exists(file_path):
                    raise ProcessorError(f"File not found: {file_path}")

                if not self.validator.validate_file_size(file_path, max_size_mb=100):
                    raise ProcessorError(f"File too large: {file_path}")

                # Extract HTML content
                content, metadata = self._extract_html_content(file_path, config)

                self.logger.info(
                    f"Extracted HTML content",
                    file_path=file_path,
                    content_length=len(content),
                    title=metadata.get('title', 'N/A'),
                )

                return {'content': content, 'metadata': metadata}

        except Exception as e:
            error_msg = self.error_handler.handle_file_error(item.name, e)
            raise ProcessorError(error_msg)

    def _extract_html_content(self, file_path: str, config: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Extract content from HTML file.

        Args:
            file_path: Path to the HTML file
            config: Processing configuration

        Returns:
            Tuple of (content, metadata)
        """
        try:
            from bs4 import BeautifulSoup

            # Read HTML file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()

            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')

            # Extract metadata
            metadata = self._extract_html_metadata(soup, config)

            # Extract text content
            if config.get('preserve_structure', True):
                content = self._extract_structured_text(soup, config)
            else:
                content = self._extract_plain_text(soup)

            return content, metadata

        except ImportError:
            # Fallback to basic HTML parsing if BeautifulSoup is not available
            self.logger.warning("BeautifulSoup not available, using basic HTML parsing")
            return self._extract_html_basic(file_path, config)
        except Exception as e:
            self.logger.error(f"Failed to process HTML file: {e}")
            raise ProcessorError(f"HTML processing failed: {str(e)}")

    def _extract_html_metadata(self, soup: BeautifulSoup, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from HTML.

        Args:
            soup: BeautifulSoup object
            config: Processing configuration

        Returns:
            Dictionary with HTML metadata
        """
        metadata = {'file_type': 'html', 'title': '', 'description': '', 'links': [], 'images': []}

        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()

        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc:
            metadata['description'] = meta_desc.get('content', '').strip()

        # Extract links if requested
        if config.get('extract_links', True):
            links = []
            for link in soup.find_all('a', href=True):
                links.append({'text': link.get_text().strip(), 'url': link.get('href')})
            metadata['links'] = links

        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            images.append({'alt': img.get('alt', ''), 'src': img.get('src')})
        metadata['images'] = images

        return metadata

    def _extract_structured_text(self, soup: BeautifulSoup, config: Dict[str, Any]) -> str:
        """
        Extract structured text from HTML.

        Args:
            soup: BeautifulSoup object
            config: Processing configuration

        Returns:
            Structured text content
        """
        lines = []

        # Extract title
        title = soup.find('title')
        if title:
            lines.append(f"Title: {title.get_text().strip()}")
            lines.append("")

        # Extract headings and content
        for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div']):
            tag_name = element.name
            text = element.get_text().strip()

            if not text:
                continue

            if tag_name.startswith('h'):
                # Heading
                level = int(tag_name[1])
                lines.append(f"{'#' * level} {text}")
            elif tag_name == 'p':
                # Paragraph
                lines.append(text)
            elif tag_name == 'div':
                # Div content
                lines.append(text)

        return "\n".join(lines)

    def _extract_plain_text(self, soup: BeautifulSoup) -> str:
        """
        Extract plain text from HTML.

        Args:
            soup: BeautifulSoup object

        Returns:
            Plain text content
        """
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)

        return text

    def _extract_html_basic(self, file_path: str, config: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Basic HTML extraction without BeautifulSoup.

        Args:
            file_path: Path to the HTML file
            config: Processing configuration

        Returns:
            Tuple of (content, metadata)
        """
        import re

        # Read HTML file
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            html_content = f.read()

        # Extract title
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ''

        # Extract meta description
        desc_match = re.search(
            r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']', html_content, re.IGNORECASE
        )
        description = desc_match.group(1).strip() if desc_match else ''

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        metadata = {'file_type': 'html', 'title': title, 'description': description, 'links': [], 'images': []}

        return text, metadata

    def _get_processor_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get processor-specific metadata.

        Args:
            config: Processing configuration

        Returns:
            Dictionary with processor-specific metadata
        """
        metadata = super()._get_processor_metadata(config)
        metadata.update(
            {
                'processor_type': 'html',
                'preserve_structure': config.get('preserve_structure', True),
                'extract_links': config.get('extract_links', True),
            }
        )
        return metadata
