"""
Example usage of the multi-MIME type processor system.
"""

import dtlpy as dl
from pipeline.processor_factory import ProcessorFactory, process_document


def example_basic_usage():
    """Example of basic processor usage."""
    print("=== Basic Processor Usage ===")

    # Get supported MIME types
    supported_types = ProcessorFactory.get_supported_mime_types()
    print(f"Supported MIME types: {supported_types}")

    # Create processors for different file types
    text_processor = ProcessorFactory.create_processor('text/plain')
    html_processor = ProcessorFactory.create_processor('text/html')
    pdf_processor = ProcessorFactory.create_processor('application/pdf')
    email_processor = ProcessorFactory.create_processor('message/rfc822')

    print(f"Text processor: {text_processor.processor_type}")
    print(f"HTML processor: {html_processor.processor_type}")
    print(f"PDF processor: {pdf_processor.processor_type}")
    print(f"Email processor: {email_processor.processor_type}")


def example_dataloop_integration():
    """Example of integrating with Dataloop."""
    print("\n=== Dataloop Integration Example ===")

    # Mock Dataloop components for demonstration
    class MockItem:
        def __init__(self, name, mimetype):
            self.name = name
            self.mimetype = mimetype
            self.id = "mock_item_123"
            self.dataset = MockDataset()

        def download(self, local_path):
            return f"/tmp/{self.name}"

    class MockDataset:
        def __init__(self):
            self.name = "test_dataset"

    class MockContext:
        def __init__(self):
            self.node = MockNode()

    class MockNode:
        def __init__(self):
            self.metadata = {
                'customNodeConfig': {
                    'chunking_strategy': 'recursive',
                    'max_chunk_size': 300,
                    'chunk_overlap': 20,
                    'to_correct_spelling': False,
                }
            }

    # Create mock components
    mock_item = MockItem("document.pdf", "application/pdf")
    mock_dataset = MockDataset()
    mock_context = MockContext()

    # Process document
    try:
        result = process_document(mock_item, mock_dataset, mock_context)
        print(f"Processing completed: {len(result)} chunks created")
    except Exception as e:
        print(f"Processing failed: {e}")


def example_custom_configuration():
    """Example of using custom configuration."""
    print("\n=== Custom Configuration Example ===")

    # Create processor with custom config
    processor = ProcessorFactory.create_processor('text/plain')

    # Mock context with custom configuration
    class MockContext:
        def __init__(self):
            self.node = MockNode()

    class MockNode:
        def __init__(self):
            self.metadata = {
                'customNodeConfig': {
                    'chunking_strategy': 'fixed-size',
                    'max_chunk_size': 500,
                    'chunk_overlap': 50,
                    'to_correct_spelling': True,
                    'detect_encoding': True,
                    'preserve_csv_structure': True,
                }
            }

    mock_context = MockContext()
    config = processor._get_config(mock_context)

    print("Custom configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")


def example_error_handling():
    """Example of error handling."""
    print("\n=== Error Handling Example ===")

    # Test unsupported MIME type
    processor = ProcessorFactory.create_processor('application/unsupported')
    if processor is None:
        print("Unsupported MIME type handled gracefully")

    # Test processor creation
    try:
        processor = ProcessorFactory.create_processor('text/plain')
        print(f"Processor created successfully: {processor.processor_type}")
    except Exception as e:
        print(f"Error creating processor: {e}")


def example_processor_registry():
    """Example of processor registry functionality."""
    print("\n=== Processor Registry Example ===")

    # Get processor information
    processor_info = ProcessorFactory.get_processor_info()
    print("Available processors:")
    for mime_type, description in processor_info.items():
        print(f"  {mime_type}: {description}")

    # Check if MIME type is supported
    test_types = ['text/plain', 'application/pdf', 'unsupported/type']
    for mime_type in test_types:
        is_supported = ProcessorFactory.is_supported(mime_type)
        print(f"  {mime_type}: {'Supported' if is_supported else 'Not supported'}")


def example_file_processing():
    """Example of processing files by extension."""
    print("\n=== File Processing Example ===")

    from pipeline.processor_factory import get_processor_for_file

    # Test different file types
    test_files = ["document.txt", "data.csv", "page.html", "email.eml", "report.pdf", "unknown.xyz"]

    for file_path in test_files:
        processor = get_processor_for_file(file_path)
        if processor:
            print(f"  {file_path}: {processor.processor_type} processor")
        else:
            print(f"  {file_path}: No processor available")


def main():
    """Run all examples."""
    print("Multi-MIME Type Processor System Examples")
    print("=" * 50)

    example_basic_usage()
    example_dataloop_integration()
    example_custom_configuration()
    example_error_handling()
    example_processor_registry()
    example_file_processing()

    print("\n" + "=" * 50)
    print("Examples completed successfully!")


if __name__ == "__main__":
    main()


