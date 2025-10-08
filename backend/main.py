from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import json
import asyncio
from typing import Optional
import logging
from pathlib import Path

# Import your existing DocumentProcessor class
from auditor import DocumentProcessor
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Processing API",
    description="API for processing Form D, Invoice, and BL (Bill of Lading) PDF documents using AI",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global processor instance
processor = None

@app.on_event("startup")
async def startup_event():
    """Initialize the document processor on startup"""
    global processor
    try:
        # Initialize with environment variable for API key
        processor = DocumentProcessor()
        logger.info("Document processor initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize document processor: {e}")
        raise e

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Document Processing API is running",
        "status": "healthy",
        "version": "2.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "processor_initialized": processor is not None,
        "api_version": "2.0.0"
    }

@app.post("/process-documents")
async def process_documents(
    formd_pdf: UploadFile = File(..., description="Form D PDF file"),
    invoice_pdf: UploadFile = File(..., description="Invoice PDF file"),
    bl_pdf: UploadFile = File(..., description="BL (Bill of Lading) PDF file"),
    formd_page: int = Form(0, description="Page number for Form D (0-indexed)"),
    invoice_page: int = Form(2, description="Page number for Invoice (0-indexed)"),
    bl_page: int = Form(0, description="Page number for BL (0-indexed)"),
    rotate_invoice: bool = Form(True, description="Whether to rotate invoice image 90° clockwise"),
    crop_bottom_percent: float = Form(30.0, description="Percentage of bottom to crop from invoice and BL images"),
    api_key: Optional[str] = Form(None, description="Optional OpenAI API key override")
):
    """
    Process Form D, Invoice, and BL PDF documents
    
    - **formd_pdf**: Upload the Form D PDF file
    - **invoice_pdf**: Upload the Invoice PDF file
    - **bl_pdf**: Upload the BL (Bill of Lading) PDF file
    - **formd_page**: Page number to process from Form D (default: 0)
    - **invoice_page**: Page number to process from Invoice (default: 2)
    - **bl_page**: Page number to process from BL (default: 0)
    - **rotate_invoice**: Rotate invoice image 90° clockwise (default: true)
    - **crop_bottom_percent**: Percentage of bottom to crop from invoice and BL (default: 30%)
    - **api_key**: Optional OpenAI API key override
    """
    
    if processor is None:
        raise HTTPException(status_code=500, detail="Document processor not initialized")
    
    # Validate file types
    if not formd_pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Form D file must be a PDF")
    
    if not invoice_pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invoice file must be a PDF")
    
    if not bl_pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="BL file must be a PDF")
    
    # Validate parameters
    if formd_page < 0:
        raise HTTPException(status_code=400, detail="Form D page number must be >= 0")
    
    if invoice_page < 0:
        raise HTTPException(status_code=400, detail="Invoice page number must be >= 0")
    
    if bl_page < 0:
        raise HTTPException(status_code=400, detail="BL page number must be >= 0")
    
    if crop_bottom_percent < 0 or crop_bottom_percent > 50:
        raise HTTPException(status_code=400, detail="Crop bottom percentage must be between 0 and 50")
    
    # Create temporary files
    temp_dir = None
    formd_temp_path = None
    invoice_temp_path = None
    bl_temp_path = None
    
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded files to temporary location
        formd_temp_path = os.path.join(temp_dir, f"formd_{formd_pdf.filename}")
        invoice_temp_path = os.path.join(temp_dir, f"invoice_{invoice_pdf.filename}")
        bl_temp_path = os.path.join(temp_dir, f"bl_{bl_pdf.filename}")
        
        # Write uploaded files
        with open(formd_temp_path, "wb") as f:
            content = await formd_pdf.read()
            f.write(content)
        
        with open(invoice_temp_path, "wb") as f:
            content = await invoice_pdf.read()
            f.write(content)
        
        with open(bl_temp_path, "wb") as f:
            content = await bl_pdf.read()
            f.write(content)
        
        logger.info(f"Processing documents: {formd_pdf.filename}, {invoice_pdf.filename}, and {bl_pdf.filename}")
        logger.info(f"Parameters: formd_page={formd_page}, invoice_page={invoice_page}, bl_page={bl_page}, "
                   f"rotate_invoice={rotate_invoice}, crop_bottom_percent={crop_bottom_percent}")
        
        # Create processor instance with custom API key if provided
        current_processor = processor
        if api_key:
            current_processor = DocumentProcessor(api_key=api_key)
        
        # Process documents
        result = current_processor.process_documents(
            formd_pdf_path=formd_temp_path,
            invoice_pdf_path=invoice_temp_path,
            bl_pdf_path=bl_temp_path,
            formd_page=formd_page,
            invoice_page=invoice_page,
            bl_page=bl_page,
            rotate_invoice=rotate_invoice,
            crop_bottom_percent=crop_bottom_percent
        )
        
        # Check if there was an error in processing
        if "error" in result:
            raise HTTPException(status_code=500, detail=f"Processing error: {result['error']}")
        
        logger.info("Document processing completed successfully")
        
        # Return the result
        return JSONResponse(content={
            "success": True,
            "message": "Documents processed successfully",
            "data": result,
            "metadata": {
                "formd_filename": formd_pdf.filename,
                "invoice_filename": invoice_pdf.filename,
                "bl_filename": bl_pdf.filename,
                "formd_page": formd_page,
                "invoice_page": invoice_page,
                "bl_page": bl_page,
                "rotate_invoice": rotate_invoice,
                "crop_bottom_percent": crop_bottom_percent
            }
        })
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up temporary files
        try:
            if formd_temp_path and os.path.exists(formd_temp_path):
                os.remove(formd_temp_path)
            if invoice_temp_path and os.path.exists(invoice_temp_path):
                os.remove(invoice_temp_path)
            if bl_temp_path and os.path.exists(bl_temp_path):
                os.remove(bl_temp_path)
            if temp_dir and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {e}")

@app.post("/process-documents-simple")
async def process_documents_simple(
    formd_pdf: UploadFile = File(..., description="Form D PDF file"),
    invoice_pdf: UploadFile = File(..., description="Invoice PDF file"),
    bl_pdf: UploadFile = File(..., description="BL (Bill of Lading) PDF file")
):
    """
    Simplified endpoint for processing documents with default parameters
    
    - **formd_pdf**: Upload the Form D PDF file
    - **invoice_pdf**: Upload the Invoice PDF file
    - **bl_pdf**: Upload the BL (Bill of Lading) PDF file
    """
    return await process_documents(
        formd_pdf=formd_pdf,
        invoice_pdf=invoice_pdf,
        bl_pdf=bl_pdf,
        formd_page=0,
        invoice_page=2,
        bl_page=0,
        rotate_invoice=True,
        crop_bottom_percent=30.0,
        api_key=None
    )

@app.get("/api-info")
async def api_info():
    """Get information about the API and available endpoints"""
    return {
        "title": "Document Processing API",
        "version": "2.0.0",
        "description": "API for processing Form D, Invoice, and BL (Bill of Lading) PDF documents using AI",
        "endpoints": {
            "/": "Root endpoint - health check",
            "/health": "Detailed health check",
            "/process-documents": "Main endpoint for document processing with full parameters",
            "/process-documents-simple": "Simplified endpoint with default parameters",
            "/api-info": "This endpoint - API information",
            "/docs": "Interactive API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation (ReDoc)"
        },
        "supported_formats": ["PDF"],
        "features": [
            "AI-powered text extraction from PDF documents",
            "Three-way document comparison (Form D, Invoice, and BL)",
            "Document relationship analysis with confidence scoring",
            "Validation against product and company databases",
            "Configurable image processing (rotation, cropping)",
            "Detailed confidence scoring and recommendations"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set.")
        print("You can either:")
        print("1. Set the environment variable: export OPENAI_API_KEY='your-api-key'")
        print("2. Pass the API key in the request body")
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )