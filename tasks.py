from celery import Celery
from celery.utils.log import get_task_logger
from database import BUCKET_NAME, s3_client, supabase
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.docx import partition_docx
from unstructured.partition.html import partition_html
import os
import tempfile

logger = get_task_logger(__name__)


# Create Celery app
celery_app = Celery(
    'document_processor', #Name of our Celery app
    broker="redis://localhost:6379/0", # Where tasks are queued
    backend="redis://localhost:6379/0" # Where results are stored 
)


def update_status(document_id: str, status: str, details: dict = None):
    """ Update document processing status with optional details """

    # Get current document 
    result = supabase.table("project_documents").select("processing_details").eq("id", document_id).execute()

    # Start with existing details or empty dict
    current_details = {}

    if result.data and result.data[0]["processing_details"]:
        current_details = result.data[0]["processing_details"]

    
    # Add new details if provided
    if details: 
        current_details.update(details)
    

    # Update document 
    supabase.table("project_documents").update({
        "processing_status": status, 
        "processing_details": current_details
    }).eq("id", document_id).execute()


@celery_app.task
def process_document(document_id: str):
    """
        Real document Processing
    """

    try: 

        doc_result = supabase.table("project_documents").select("*").eq("id", document_id).execute()
        document = doc_result.data[0]

        # step 1: Download and partition 
        update_status(document_id, "partitioning")
        elements = download_and_partition(document_id, document)

        tables = sum(1 for e in elements if e.category == "Table")
        images = sum(1 for e in elements if e.category == "Image")
        text_elements = sum(1 for e in elements if e.category in ["NarrativeText", "Title", "Text"])
        logger.info(f"📊 Extracted: {tables} tables, {images} images, {text_elements} text elements")


        # step 2: Chunk elements 

        #3 Step 3: Summarising chunks 

        #4 Step 4: Vectorization & storing 
        
    
        return {
            "status": "success", 
            "document_id": document_id
        }

    except Exception as e:
        logger.error(f"❌ Failed to process document {document_id}: {str(e)}", exc_info=True)
        update_status(document_id, "failed", {"error": str(e)})
        raise  # Re-raise so Celery marks task as FAILED
   

def download_and_partition(document_id: str, document: dict):
    """ Download document from S3 / Crawl URL and partition into elements  """
    
    logger.info(f"⬇️ Downloading and partitioning document {document_id}")

    source_type = document.get("source_type", "file")

    if source_type == "url":
        # Crawl URL 
        pass

    else:
        # Handle file processing
        
        s3_key = document["s3_key"]
        filename = document["filename"]
        file_type = filename.split(".")[-1].lower()

        #  Download to a temporary location 
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"{document_id}.{file_type}")
        logger.info(f"📥 Downloading from S3: {s3_key}")

        s3_client.download_file(BUCKET_NAME, s3_key, temp_file)
        logger.info(f"✅ Downloaded to {temp_file}")

    try:
        elements = partition_document(temp_file, file_type, source_type="file")
        logger.info(f"✅ Partitioned document into {len(elements)} elements")
    finally:
        # ✅ Fix: Always clean up temp file even if partitioning fails
            if os.path.exists(temp_file):
                os.remove(temp_file)
                logger.info(f"🗑️ Cleaned up temp file")

    elements_summary = analyze_elements(elements)
    logger.info(f"📊 Elements summary: {elements_summary}")

    update_status(document_id, "chunking", {
        "partitioning": {
            "elements_found": elements_summary
        }
    })


    return elements



def partition_document(temp_file: str, file_type: str, source_type: str = "file"):
    """ Partition document based on file type and source type """

    logger.info(f"🔍 Partitioning {file_type} file...")

    if source_type == "url": 
        pass

    if file_type == "pdf":
        return partition_pdf(
            filename=temp_file,  # Path to your PDF file
            strategy="hi_res", # Use the most accurate (but slower) processing method of extraction
            infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"], # Grab images found in the PDF
            extract_image_block_to_payload=True # Store images as base64 data you can actually use
        )



def analyze_elements(elements):
    """ Count different types of elements found in the document """

    text_count = 0
    table_count = 0
    image_count = 0
    title_count = 0
    other_count = 0

    # Go through each element and count what type it is 
    for element in elements: 
        element_name = type(element).__name__ #Get the class name like "Table" or "NarrativeText"

        if element_name == "Table":
            table_count += 1
        elif element_name == "Image": 
            image_count += 1
        elif element_name in ["Title", "Header"]:
            title_count += 1
        elif element_name in ["NarrativeText", "Text", "ListItem", "FigureCaption"]:
            text_count += 1
        else:
            other_count += 1

    # Return a simple dictionary
    return {
        "text": text_count,
        "tables": table_count,
        "images": image_count,
        "titles": title_count,
        "other": other_count
    }
