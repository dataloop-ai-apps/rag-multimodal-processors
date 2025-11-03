"""
Email processor for handling email files (.eml).
"""

import os
import email
from email.header import decode_header
from typing import Dict, Any, List
import dtlpy as dl
from pipeline.base.processor import BaseProcessor, ProcessorError
from pipeline.utils.logging_utils import ProcessorLogger, ErrorHandler, FileValidator


class EmailProcessor(BaseProcessor):
    """
    Processor for email files (.eml).
    """

    def __init__(self):
        """Initialize email processor."""
        super().__init__('email')
        self.logger = ProcessorLogger('email')
        self.error_handler = ErrorHandler('email')
        self.validator = FileValidator()

    def _extract_content(self, item: dl.Item, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract content from email file.

        Args:
            item: Email file item
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

                # Extract email content
                content, metadata = self._extract_email_content(file_path, config)

                self.logger.info(
                    f"Extracted email content",
                    file_path=file_path,
                    content_length=len(content),
                    subject=metadata.get('subject', 'N/A'),
                )

                return {'content': content, 'metadata': metadata}

        except Exception as e:
            error_msg = self.error_handler.handle_file_error(item.name, e)
            raise ProcessorError(error_msg)

    def _extract_email_content(self, file_path: str, config: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Extract content from email file.

        Args:
            file_path: Path to the email file
            config: Processing configuration

        Returns:
            Tuple of (content, metadata)
        """
        try:
            # Read email file
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                email_content = f.read()

            # Parse email
            msg = email.message_from_string(email_content)

            # Extract metadata
            metadata = self._extract_email_metadata(msg, config)

            # Extract text content
            content = self._extract_email_text(msg, config)

            return content, metadata

        except Exception as e:
            self.logger.error(f"Failed to process email file: {e}")
            raise ProcessorError(f"Email processing failed: {str(e)}")

    def _extract_email_metadata(self, msg: email.message.Message, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from email.

        Args:
            msg: Email message object
            config: Processing configuration

        Returns:
            Dictionary with email metadata
        """
        metadata = {'file_type': 'email', 'from': '', 'to': '', 'subject': '', 'date': '', 'attachments': []}

        # Extract headers
        if config.get('extract_headers', True):
            metadata['from'] = self._decode_header(msg.get('From', ''))
            metadata['to'] = self._decode_header(msg.get('To', ''))
            metadata['subject'] = self._decode_header(msg.get('Subject', ''))
            metadata['date'] = self._decode_header(msg.get('Date', ''))

        # Extract attachments if requested
        if config.get('include_attachments', False):
            attachments = []
            for part in msg.walk():
                if part.get_content_disposition() == 'attachment':
                    filename = part.get_filename()
                    if filename:
                        attachments.append(self._decode_header(filename))
            metadata['attachments'] = attachments

        return metadata

    def _extract_email_text(self, msg: email.message.Message, config: Dict[str, Any]) -> str:
        """
        Extract text content from email.

        Args:
            msg: Email message object
            config: Processing configuration

        Returns:
            Text content
        """
        lines = []

        # Add headers if requested
        if config.get('extract_headers', True):
            lines.append("Email Headers:")
            lines.append("=" * 50)
            lines.append(f"From: {self._decode_header(msg.get('From', ''))}")
            lines.append(f"To: {self._decode_header(msg.get('To', ''))}")
            lines.append(f"Subject: {self._decode_header(msg.get('Subject', ''))}")
            lines.append(f"Date: {self._decode_header(msg.get('Date', ''))}")
            lines.append("")

        # Extract body content
        lines.append("Email Body:")
        lines.append("=" * 50)

        # Get the main text content
        body_text = self._get_email_body(msg)
        if body_text:
            lines.append(body_text)
        else:
            lines.append("No text content found")

        return "\n".join(lines)

    def _get_email_body(self, msg: email.message.Message) -> str:
        """
        Get the main text body from email.

        Args:
            msg: Email message object

        Returns:
            Email body text
        """
        body_text = ""

        if msg.is_multipart():
            # Handle multipart messages
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get("Content-Disposition"))

                # Skip attachments
                if "attachment" in content_disposition:
                    continue

                # Get text content
                if content_type == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        try:
                            charset = part.get_content_charset() or 'utf-8'
                            body_text = payload.decode(charset, errors='replace')
                            break
                        except Exception as e:
                            self.logger.warning(f"Failed to decode text part: {e}")
                            continue
                elif content_type == "text/html":
                    # Use HTML as fallback if no plain text
                    if not body_text:
                        payload = part.get_payload(decode=True)
                        if payload:
                            try:
                                charset = part.get_content_charset() or 'utf-8'
                                html_content = payload.decode(charset, errors='replace')
                                # Basic HTML tag removal
                                import re

                                body_text = re.sub(r'<[^>]+>', '', html_content)
                                body_text = re.sub(r'\s+', ' ', body_text).strip()
                            except Exception as e:
                                self.logger.warning(f"Failed to decode HTML part: {e}")
                                continue
        else:
            # Handle single part messages
            content_type = msg.get_content_type()
            if content_type in ["text/plain", "text/html"]:
                payload = msg.get_payload(decode=True)
                if payload:
                    try:
                        charset = msg.get_content_charset() or 'utf-8'
                        body_text = payload.decode(charset, errors='replace')

                        # Remove HTML tags if it's HTML
                        if content_type == "text/html":
                            import re

                            body_text = re.sub(r'<[^>]+>', '', body_text)
                            body_text = re.sub(r'\s+', ' ', body_text).strip()
                    except Exception as e:
                        self.logger.warning(f"Failed to decode email body: {e}")

        return body_text

    def _decode_header(self, header: str) -> str:
        """
        Decode email header.

        Args:
            header: Header string to decode

        Returns:
            Decoded header string
        """
        if not header:
            return ""

        try:
            decoded_parts = decode_header(header)
            decoded_string = ""

            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        decoded_string += part.decode(encoding, errors='replace')
                    else:
                        decoded_string += part.decode('utf-8', errors='replace')
                else:
                    decoded_string += part

            return decoded_string.strip()
        except Exception as e:
            self.logger.warning(f"Failed to decode header: {e}")
            return header

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
                'processor_type': 'email',
                'extract_headers': config.get('extract_headers', True),
                'include_attachments': config.get('include_attachments', False),
            }
        )
        return metadata
