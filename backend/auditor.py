import json
import pandas as pd
from pdf2image import convert_from_path
import sys
import chardet
from typing import Dict, Any, Tuple, List
import difflib
import os
import re
import time
import base64
from io import BytesIO
from openai import OpenAI
from PIL import Image

class DocumentProcessor:
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """Initialize the document processor with OpenAI API"""
        print("Initializing OpenAI client...")
        
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = OpenAI()
        
        self.model = model
        print(f"Using OpenAI model: {self.model}")
        
        self.load_validation_data()
        
    def detect_encoding(self, file_path: str) -> str:
        """Detect the encoding of a file"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                print(f"Detected encoding for {file_path}: {encoding} (confidence: {confidence:.2f})")
                return encoding if confidence > 0.7 else 'utf-8'
        except Exception as e:
            print(f"Could not detect encoding for {file_path}: {e}")
            return 'utf-8'
    
    def load_csv_with_fallback(self, file_path: str) -> pd.DataFrame:
        """Load CSV with multiple encoding fallbacks"""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
            
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        detected_encoding = self.detect_encoding(file_path)
        if detected_encoding and detected_encoding not in encodings:
            encodings.insert(0, detected_encoding)
        
        for encoding in encodings:
            try:
                print(f"Trying to load {file_path} with encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"Successfully loaded {file_path} with {encoding} encoding")
                print(f"Shape: {df.shape}, Columns: {list(df.columns)}")
                return df
            except UnicodeDecodeError as e:
                print(f"Failed to load with {encoding}: {e}")
                continue
            except Exception as e:
                print(f"Error loading {file_path} with {encoding}: {e}")
                continue
        
        print(f"Failed to load {file_path} with any encoding")
        return None

    def load_validation_data(self):
        """Load CSV data for validation"""
        try:
            product_file = './data/5.1 Abbott IR H.S code and FormD desctiption (14-05-24) updated 13Aug 24.csv'
            self.product_data = self.load_csv_with_fallback(product_file)
            
            company_file = './data/Companies.csv'
            self.company_data = self.load_csv_with_fallback(company_file)
            
            abbott_file = './data/Abbott.csv'
            self.abbott_data = self.load_csv_with_fallback(abbott_file)
            
            if any([self.product_data is not None, self.company_data is not None, self.abbott_data is not None]):
                print("Validation data loaded successfully")
            else:
                print("Warning: No validation data could be loaded")
                
        except Exception as e:
            print(f"Warning: Could not load validation data: {e}")
            self.product_data = None
            self.company_data = None
            self.abbott_data = None

    def image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str

    def extract_image_info(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Extract information from an image using OpenAI Vision API"""
        try:
            base64_image = self.image_to_base64(image)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            output_text = response.choices[0].message.content
            
            try:
                return json.loads(output_text)
            except json.JSONDecodeError:
                print("Warning: Could not parse JSON response, attempting to clean...")
                json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
                
                print(f"Raw response: {output_text}")
                return {"raw_response": output_text, "parsing_error": "Could not parse as JSON"}
                
        except Exception as e:
            print(f"Error in extract_image_info: {e}")
            return {"error": str(e)}

    def extract_formd_info(self, formd_pdf_path: str, page_numbers: List[int] = None, save_debug_image: bool = False) -> Dict[str, Any]:
        """Extract information from specified pages of Form D document"""
        try:
            print(f"Converting Form D PDF to images...")
            images = convert_from_path(formd_pdf_path, dpi=300)
            print(f"Form D has {len(images)} page(s)")
            
            # If no page numbers specified, use only first page
            if page_numbers is None:
                page_numbers = [0]
            
            # Validate page numbers
            valid_pages = []
            for page_num in page_numbers:
                if 0 <= page_num < len(images):
                    valid_pages.append(page_num)
                else:
                    print(f"Warning: Page {page_num} is out of range (0-{len(images)-1}). Skipping.")
            
            if not valid_pages:
                raise ValueError("No valid pages to process")
            
            print(f"Processing Form D pages: {valid_pages}")
            
            all_products = []
            common_info = {}
            
            for idx, page_index in enumerate(valid_pages):
                print(f"\nProcessing Form D page {page_index} ({idx + 1}/{len(valid_pages)})...")
                image = images[page_index]
                
                if save_debug_image:
                    debug_filename = f"debug_formd_page{page_index}.png"
                    image.save(debug_filename)
                    print(f"Debug image saved: {debug_filename}")
                
                if idx == 0:
                    # First selected page: extract common info + products
                    question = """Extract ALL information from this Form D page as JSON:
    {
    "Exporter's business name": "...",
    "Exporter's address": "...",
    "Exporter's country": "...",
    "Consignee's business name": "...",
    "Consignee's address": "...",
    "Consignee's country": "...",
    "products": [
        {
        "Item Number": "...",
        "HS CODE": "complete float number HS CODE",
        "Product Description": "...",
        "CTNS": "...",
        "Gross weight": "...",
        "Number of invoices": "...",
        "Date of invoices": "..."
        }
    ]
    }

    Important:
    - Extract ALL product items from the table (items 1, 2, 3, 4, 5, etc.)
    - Each row in columns 5-10 is a separate product
    - If information is not found, use "Not found"
    - Return only valid JSON"""
                else:
                    # Subsequent pages: extract only products
                    question = """Extract ALL product items from this Form D continuation page as JSON:
    {
    "products": [
        {
        "Item Number": "...",
        "HS CODE": "float number HS CODE",
        "Product Description": "...",
        "CTNS": "...",
        "Gross weight": "...",
        "Number of invoices": "...",
        "Date of invoices": "..."
        }
    ]
    }

    Important:
    - Extract ALL product items from the table
    - Each row is a separate product
    - If information is not found, use "Not found"
    - Return only valid JSON"""
                
                page_data = self.extract_image_info(image, question)
                
                if idx == 0 and "products" in page_data:
                    # Save common info from first selected page
                    common_info = {k: v for k, v in page_data.items() if k != "products"}
                    all_products.extend(page_data["products"])
                elif "products" in page_data:
                    all_products.extend(page_data["products"])
            
            # Combine common info with all products
            result = common_info.copy()
            result["products"] = all_products
            result["total_products"] = len(all_products)
            result["pages_processed"] = valid_pages
            
            print(f"\nExtracted {len(all_products)} product(s) from Form D pages {valid_pages}")
            return result
            
        except Exception as e:
            print(f"Error extracting Form D info: {e}")
            return {"error": str(e)}

    def extract_bl_info(self, bl_pdf_path: str, page_numbers: List[int] = None, crop_bottom_percent: float = 18, save_debug_image: bool = False) -> Dict[str, Any]:
        """Extract information from specified pages of BL document"""
        try:
            print(f"Converting BL PDF to images...")
            images = convert_from_path(bl_pdf_path, dpi=300)
            print(f"BL has {len(images)} page(s)")
            
            # If no page numbers specified, use only first page
            if page_numbers is None:
                page_numbers = [0]
            
            # Validate page numbers
            valid_pages = []
            for page_num in page_numbers:
                if 0 <= page_num < len(images):
                    valid_pages.append(page_num)
                else:
                    print(f"Warning: Page {page_num} is out of range (0-{len(images)-1}). Skipping.")
            
            if not valid_pages:
                raise ValueError("No valid pages to process")
            
            print(f"Processing BL pages: {valid_pages}")
            
            all_containers = []
            common_info = {}
            
            for idx, page_index in enumerate(valid_pages):
                print(f"\nProcessing BL page {page_index} ({idx + 1}/{len(valid_pages)})...")
                image = images[page_index]
                
                if crop_bottom_percent > 0:
                    width, height = image.size
                    crop_pixels = int(height * crop_bottom_percent / 100)
                    cropped_height = height - crop_pixels
                    print(f"Cropping {crop_bottom_percent}% ({crop_pixels} pixels) from bottom. New height: {cropped_height}")
                    crop_box = (0, 0, width, cropped_height)
                    image = image.crop(crop_box)
                
                if save_debug_image:
                    debug_filename = f"debug_bl_page{page_index}_cropped.png"
                    image.save(debug_filename)
                    print(f"Debug image saved: {debug_filename}")
                
                # Always ask for ALL fields from each page - let the model return what's available
                question = """Extract ALL information from this Bill of Lading page as JSON:
    {
    "shipper_name": "...",
    "shipper_address": "...",
    "consignee_name": "...",
    "consignee_address": "...",
    "notify_party_name": "...",
    "notify_party_address": "...",
    "port_of_loading": "...",
    "port_of_discharge": "...",
    "shipment_number": "...",
    "number_of_containers": "...",
    "number_of_cartons": "...",
    "total_number_of_packages": "total if shown",
    "total_gross_weight": "total if shown",
    "containers": [
        {
        "description_of_goods": "...",
        "carton": "...",
        "gross_weight": "...",
        }
    ]
    }

    Important:
    - Extract ALL available information from this page
    - For header/summary fields not present on this page, use "Not found"
    - Extract ALL container entries from the table if present
    - If information is not found, use "Not found"
    - Return only valid JSON"""
                
                page_data = self.extract_image_info(image, question)
                
                # Merge common info from all pages
                if idx == 0:
                    # First page: initialize common_info
                    common_info = {k: v for k, v in page_data.items() if k != "containers"}
                else:
                    # Subsequent pages: merge non-container fields
                    # Update common_info with any non-empty values from this page
                    for key, value in page_data.items():
                        if key != "containers":
                            # Update if current value is "Not found" or empty, and new value is not
                            current_val = common_info.get(key, "")
                            if (not current_val or current_val == "Not found") and value and value != "Not found":
                                common_info[key] = value
                                print(f"  Updated '{key}' from page {page_index}: {value}")
                
                # Collect containers from all pages
                if "containers" in page_data:
                    all_containers.extend(page_data["containers"])
            
            # Combine common info with all containers
            result = common_info.copy()
            result["containers"] = all_containers
            result["total_containers"] = len(all_containers)
            result["pages_processed"] = valid_pages
            
            print(f"\nExtracted {len(all_containers)} container(s) from BL pages {valid_pages}")
            return result
            
        except Exception as e:
            print(f"Error extracting BL info: {e}")
            return {"error": str(e)}


    def extract_invoice_info(self, invoice_pdf_path: str, page_index: int = 2, rotate_image: bool = True, crop_bottom_percent: float = 18, save_debug_image: bool = False) -> Dict[str, Any]:
        """Extract information from invoice document"""
        try:
            print(f"Converting Invoice PDF to image (page {page_index})...")
            images = convert_from_path(invoice_pdf_path, dpi=300)
            
            if page_index >= len(images):
                raise ValueError(f"Page {page_index} not found. Invoice PDF has {len(images)} pages.")
            
            image = images[page_index]
            
            if rotate_image:
                print("Rotating invoice image 90 degrees clockwise...")
                image = image.rotate(-90, expand=True)
            
            if crop_bottom_percent > 0:
                width, height = image.size
                crop_pixels = int(height * crop_bottom_percent / 100)
                cropped_height = height - crop_pixels
                print(f"Cropping {crop_bottom_percent}% ({crop_pixels} pixels) from bottom. New height: {cropped_height}")
                crop_box = (0, 0, width, cropped_height)
                image = image.crop(crop_box)
            
            if save_debug_image:
                rotation_text = "rotated_" if rotate_image else ""
                debug_filename = f"debug_invoice_page{page_index}_{rotation_text}cropped.png"
                image.save(debug_filename)
                print(f"Debug image saved: {debug_filename}")
            
            question = """Extract ALL information from the invoice as JSON:
{
  "sold_to_name": "...",
  "sold_to_address": "...",
  "ship_to_name": "...",
  "ship_to_address": "...",
  "shipped_by_name": "...",
  "shipped_by_address": "...",
  "shipment_number": "...",
  "number": "invoice number",
  "invoice_date": "...",
  "total_gross_weight_kgs": "...",
  "total_net_weight_kgs": "...",
  "packed_in": "total cartons/boxes",
  "products": [
    {
      "description": "product description",
      "lot_number": "...",
      "total_quantity": "...",
      "price_per_one": "...",
      "total_price": "..."
    }
  ]
}

Important:
- Extract ALL product lines from the table
- Include both header info and all products
- If information is not found, use "Not found"
- Return only valid JSON"""
            
            return self.extract_image_info(image, question)
            
        except Exception as e:
            print(f"Error extracting invoice info: {e}")
            return {"error": str(e)}

    def normalize_weight(self, weight_text: str) -> str:
        """Normalize weight by removing spaces, words and letters"""
        if not weight_text or weight_text == "Not found" or weight_text == "":
            return ""
        normalized = re.sub(r'[^0-9.]', '', str(weight_text))
        return normalized.strip()

    def normalize_description(self, description_text: str) -> str:
        """Normalize description by removing spaces and making all letters capital"""
        if not description_text or description_text == "Not found" or description_text == "":
            return ""
        normalized = str(description_text).replace(" ", "").upper()
        return normalized.strip()

    def normalize_text(self, text: str) -> str:
        """Normalize text for general comparison"""
        if not text or text == "Not found" or text == "":
            return ""
        return str(text).upper().strip().replace(",", "").replace(".", "").replace("-", "").replace("_", "")

    def fuzzy_match(self, text1: str, text2: str, threshold: float = 0.8) -> Tuple[bool, float]:
        """Perform fuzzy matching between two text strings"""
        if not text1 or not text2:
            return False, 0.0
        
        norm_text1 = self.normalize_text(text1)
        norm_text2 = self.normalize_text(text2)
        
        if not norm_text1 or not norm_text2:
            return False, 0.0
        
        similarity = difflib.SequenceMatcher(None, norm_text1, norm_text2).ratio()
        return similarity >= threshold, similarity

    def fuzzy_match_weight(self, weight1: str, weight2: str, threshold: float = 0.8) -> Tuple[bool, float]:
        """Perform fuzzy matching between two weight values"""
        if not weight1 or not weight2:
            return False, 0.0
        
        norm_weight1 = self.normalize_weight(weight1)
        norm_weight2 = self.normalize_weight(weight2)
        
        if not norm_weight1 or not norm_weight2:
            return False, 0.0
        
        try:
            val1 = float(norm_weight1)
            val2 = float(norm_weight2)
            if val1 == val2:
                return True, 1.0
            diff = abs(val1 - val2) / max(val1, val2) if max(val1, val2) != 0 else 0
            similarity = 1.0 - diff
            return similarity >= threshold, similarity
        except ValueError:
            similarity = difflib.SequenceMatcher(None, norm_weight1, norm_weight2).ratio()
            return similarity >= threshold, similarity

    def fuzzy_match_number(self, text1: str, text2: str, threshold: float = 0.8) -> Tuple[bool, float]:
        """Perform fuzzy matching between two texts by extracting numbers"""
        if not text1 or not text2:
            return False, 0.0
        
        num1 = re.sub(r'[^0-9.]', '', str(text1))
        num2 = re.sub(r'[^0-9.]', '', str(text2))
        
        if not num1 or not num2:
            return False, 0.0
        
        try:
            val1 = float(num1)
            val2 = float(num2)
            if val1 == val2:
                return True, 1.0
            diff = abs(val1 - val2) / max(val1, val2) if max(val1, val2) != 0 else 0
            similarity = 1.0 - diff
            return similarity >= threshold, similarity
        except ValueError:
            similarity = difflib.SequenceMatcher(None, num1, num2).ratio()
            return similarity >= threshold, similarity

    def fuzzy_match_description(self, desc1: str, desc2: str, threshold: float = 0.8) -> Tuple[bool, float]:
        """Perform fuzzy matching between two descriptions"""
        if not desc1 or not desc2:
            return False, 0.0
        
        norm_desc1 = self.normalize_description(desc1)
        norm_desc2 = self.normalize_description(desc2)
        
        if not norm_desc1 or not norm_desc2:
            return False, 0.0
        
        similarity = difflib.SequenceMatcher(None, norm_desc1, norm_desc2).ratio()
        return similarity >= threshold, similarity

    def compare_documents(self, formd_data: Dict[str, Any], invoice_data: Dict[str, Any], bl_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare Form D, invoice, and BL data - handles single and multiple products"""
        comparison_result = {
            "documents_related": False,
            "matching_fields": [],
            "discrepancies": [],
            "confidence_score": 0.0,
            "total_fields_compared": 0
        }
        
        matches = 0
        total_comparisons = 0
        
        # Get products to determine if single or multiple
        formd_products = formd_data.get("products", [])
        is_single_product = len(formd_products) == 1
        
        print(f"\nDocument comparison mode: {'SINGLE PRODUCT' if is_single_product else 'MULTIPLE PRODUCTS'}")
        print(f"Number of products in Form D: {len(formd_products)}")
        
        # Common fields for both single and multiple products
        common_field_mappings = [
            ("Consignee's business name", "ship_to_name", "consignee_name", "general"),
            ("Consignee's address", "ship_to_address", "consignee_address", "general"),
            ("Exporter's business name", "shipped_by_name", "shipper_name", "general"),
            ("Exporter's address", "shipped_by_address", "shipper_address", "general"),
        ]
        
        # Process common field comparisons (3-way)
        for formd_field, invoice_field, bl_field, comparison_type in common_field_mappings:
            if formd_field in formd_data and invoice_field in invoice_data and bl_field in bl_data:
                total_comparisons += 1
                formd_val = formd_data[formd_field]
                invoice_val = invoice_data[invoice_field]
                bl_val = bl_data[bl_field]
                
                match1, sim1 = self.fuzzy_match(str(formd_val), str(invoice_val))
                match2, sim2 = self.fuzzy_match(str(formd_val), str(bl_val))
                match3, sim3 = self.fuzzy_match(str(invoice_val), str(bl_val))
                
                avg_similarity = (sim1 + sim2 + sim3) / 3
                all_match = match1 and match2 and match3
                
                if all_match:
                    matches += 1
                    comparison_result["matching_fields"].append({
                        "field": f"{formd_field} / {invoice_field} / {bl_field}",
                        "formd_value": formd_val,
                        "invoice_value": invoice_val,
                        "bl_value": bl_val,
                        "similarity": round(avg_similarity, 3),
                        "formd_invoice_sim": round(sim1, 3),
                        "formd_bl_sim": round(sim2, 3),
                        "invoice_bl_sim": round(sim3, 3)
                    })
                elif avg_similarity > 0.1:
                    comparison_result["discrepancies"].append({
                        "field": f"{formd_field} / {invoice_field} / {bl_field}",
                        "formd_value": formd_val,
                        "invoice_value": invoice_val,
                        "bl_value": bl_val,
                        "similarity": round(avg_similarity, 3),
                        "formd_invoice_sim": round(sim1, 3),
                        "formd_bl_sim": round(sim2, 3),
                        "invoice_bl_sim": round(sim3, 3)
                    })
        
        # Handle single product scenario
        if is_single_product and formd_products:
            product = formd_products[0]
            
            # Gross weight comparison (3-way)
            formd_weight = product.get("Gross weight", "")
            invoice_weight = invoice_data.get("total_gross_weight_kgs", "")
            bl_weight = bl_data.get("total_gross_weight", "")
            
            if formd_weight and invoice_weight and bl_weight:
                total_comparisons += 1
                match1, sim1 = self.fuzzy_match_weight(formd_weight, invoice_weight)
                match2, sim2 = self.fuzzy_match_weight(formd_weight, bl_weight)
                match3, sim3 = self.fuzzy_match_weight(invoice_weight, bl_weight)
                
                avg_similarity = (sim1 + sim2 + sim3) / 3
                all_match = match1 and match2 and match3
                
                if all_match:
                    matches += 1
                    comparison_result["matching_fields"].append({
                        "field": "Gross weight / total_gross_weight_kgs / total_gross_weight",
                        "formd_value": formd_weight,
                        "invoice_value": invoice_weight,
                        "bl_value": bl_weight,
                        "similarity": round(avg_similarity, 3),
                        "formd_invoice_sim": round(sim1, 3),
                        "formd_bl_sim": round(sim2, 3),
                        "invoice_bl_sim": round(sim3, 3)
                    })
                elif avg_similarity > 0.1:
                    comparison_result["discrepancies"].append({
                        "field": "Gross weight / total_gross_weight_kgs / total_gross_weight",
                        "formd_value": formd_weight,
                        "invoice_value": invoice_weight,
                        "bl_value": bl_weight,
                        "similarity": round(avg_similarity, 3)
                    })
            
            # CTNS comparison (3-way)
            formd_ctns = product.get("CTNS", "")
            invoice_ctns = invoice_data.get("packed_in", "")
            bl_ctns = bl_data.get("number_of_cartons", "")
            
            if formd_ctns and invoice_ctns and bl_ctns:
                total_comparisons += 1
                match1, sim1 = self.fuzzy_match_number(formd_ctns, invoice_ctns)
                match2, sim2 = self.fuzzy_match_number(formd_ctns, bl_ctns)
                match3, sim3 = self.fuzzy_match_number(invoice_ctns, bl_ctns)
                
                avg_similarity = (sim1 + sim2 + sim3) / 3
                all_match = match1 and match2 and match3
                
                if all_match:
                    matches += 1
                    comparison_result["matching_fields"].append({
                        "field": "CTNS / packed_in / number_of_cartons",
                        "formd_value": formd_ctns,
                        "invoice_value": invoice_ctns,
                        "bl_value": bl_ctns,
                        "similarity": round(avg_similarity, 3),
                        "formd_invoice_sim": round(sim1, 3),
                        "formd_bl_sim": round(sim2, 3),
                        "invoice_bl_sim": round(sim3, 3)
                    })
                elif avg_similarity > 0.1:
                    comparison_result["discrepancies"].append({
                        "field": "CTNS / packed_in / number_of_cartons",
                        "formd_value": formd_ctns,
                        "invoice_value": invoice_ctns,
                        "bl_value": bl_ctns,
                        "similarity": round(avg_similarity, 3)
                    })
            
            # Number of invoices comparison (2-way: Form D vs Invoice)
            formd_num_invoices = product.get("Number of invoices", "")
            invoice_number = invoice_data.get("number", "")
            
            if formd_num_invoices and invoice_number:
                total_comparisons += 1
                is_match, similarity = self.fuzzy_match_number(formd_num_invoices, invoice_number)
                
                if is_match:
                    matches += 1
                    comparison_result["matching_fields"].append({
                        "field": "Number of invoices / number",
                        "formd_value": formd_num_invoices,
                        "invoice_value": invoice_number,
                        "similarity": round(similarity, 3)
                    })
                elif similarity > 0.3:
                    comparison_result["discrepancies"].append({
                        "field": "Number of invoices / number",
                        "formd_value": formd_num_invoices,
                        "invoice_value": invoice_number,
                        "similarity": round(similarity, 3)
                    })
            
            # Date of invoices comparison (2-way: Form D vs Invoice)
            formd_date = product.get("Date of invoices", "")
            invoice_date = invoice_data.get("invoice_date", "")
            
            if formd_date and invoice_date:
                total_comparisons += 1
                is_match, similarity = self.fuzzy_match(formd_date, invoice_date)
                
                if is_match:
                    matches += 1
                    comparison_result["matching_fields"].append({
                        "field": "Date of invoices / invoice_date",
                        "formd_value": formd_date,
                        "invoice_value": invoice_date,
                        "similarity": round(similarity, 3)
                    })
                elif similarity > 0.3:
                    comparison_result["discrepancies"].append({
                        "field": "Date of invoices / invoice_date",
                        "formd_value": formd_date,
                        "invoice_value": invoice_date,
                        "similarity": round(similarity, 3)
                    })
        
        # Handle multiple products scenario
        elif len(formd_products) > 1:
            # Total gross weight comparison (2-way: Invoice vs BL)
            invoice_weight = invoice_data.get("total_gross_weight_kgs", "")
            bl_weight = bl_data.get("total_gross_weight", "")
            
            if invoice_weight and bl_weight:
                total_comparisons += 1
                is_match, similarity = self.fuzzy_match_weight(invoice_weight, bl_weight)
                
                if is_match:
                    matches += 1
                    comparison_result["matching_fields"].append({
                        "field": "total_gross_weight_kgs / total_gross_weight",
                        "invoice_value": invoice_weight,
                        "bl_value": bl_weight,
                        "similarity": round(similarity, 3)
                    })
                elif similarity > 0.1:
                    comparison_result["discrepancies"].append({
                        "field": "total_gross_weight_kgs / total_gross_weight",
                        "invoice_value": invoice_weight,
                        "bl_value": bl_weight,
                        "similarity": round(similarity, 3)
                    })
            
            # CTNS comparison - last item from Form D (3-way)
            last_product = formd_products[-1]
            formd_ctns = last_product.get("CTNS", "")
            invoice_ctns = invoice_data.get("packed_in", "")
            bl_ctns = bl_data.get("number_of_cartons", "")
            
            if formd_ctns and invoice_ctns and bl_ctns:
                total_comparisons += 1
                match1, sim1 = self.fuzzy_match_number(formd_ctns, invoice_ctns)
                match2, sim2 = self.fuzzy_match_number(formd_ctns, bl_ctns)
                match3, sim3 = self.fuzzy_match_number(invoice_ctns, bl_ctns)
                
                avg_similarity = (sim1 + sim2 + sim3) / 3
                all_match = match1 and match2 and match3
                
                if all_match:
                    matches += 1
                    comparison_result["matching_fields"].append({
                        "field": f"CTNS (last item) / packed_in / number_of_cartons",
                        "formd_value": formd_ctns,
                        "invoice_value": invoice_ctns,
                        "bl_value": bl_ctns,
                        "similarity": round(avg_similarity, 3),
                        "note": f"Using last product (Item {last_product.get('Item Number', '?')})"
                    })
                elif avg_similarity > 0.1:
                    comparison_result["discrepancies"].append({
                        "field": f"CTNS (last item) / packed_in / number_of_cartons",
                        "formd_value": formd_ctns,
                        "invoice_value": invoice_ctns,
                        "bl_value": bl_ctns,
                        "similarity": round(avg_similarity, 3),
                        "note": f"Using last product (Item {last_product.get('Item Number', '?')})"
                    })
            
            # Number of invoices comparison - first item from Form D (2-way)
            first_product = formd_products[0]
            formd_num_invoices = first_product.get("Number of invoices", "")
            invoice_number = invoice_data.get("number", "")
            
            if formd_num_invoices and invoice_number:
                total_comparisons += 1
                is_match, similarity = self.fuzzy_match_number(formd_num_invoices, invoice_number)
                
                if is_match:
                    matches += 1
                    comparison_result["matching_fields"].append({
                        "field": "Number of invoices (first item) / number",
                        "formd_value": formd_num_invoices,
                        "invoice_value": invoice_number,
                        "similarity": round(similarity, 3),
                        "note": f"Using first product (Item {first_product.get('Item Number', '?')})"
                    })
                elif similarity > 0.3:
                    comparison_result["discrepancies"].append({
                        "field": "Number of invoices (first item) / number",
                        "formd_value": formd_num_invoices,
                        "invoice_value": invoice_number,
                        "similarity": round(similarity, 3),
                        "note": f"Using first product (Item {first_product.get('Item Number', '?')})"
                    })
            
            # Date of invoices comparison - first item from Form D (2-way)
            formd_date = first_product.get("Date of invoices", "")
            invoice_date = invoice_data.get("invoice_date", "")
            
            if formd_date and invoice_date:
                total_comparisons += 1
                is_match, similarity = self.fuzzy_match(formd_date, invoice_date)
                
                if is_match:
                    matches += 1
                    comparison_result["matching_fields"].append({
                        "field": "Date of invoices (first item) / invoice_date",
                        "formd_value": formd_date,
                        "invoice_value": invoice_date,
                        "similarity": round(similarity, 3),
                        "note": f"Using first product (Item {first_product.get('Item Number', '?')})"
                    })
                elif similarity > 0.3:
                    comparison_result["discrepancies"].append({
                        "field": "Date of invoices (first item) / invoice_date",
                        "formd_value": formd_date,
                        "invoice_value": invoice_date,
                        "similarity": round(similarity, 3),
                        "note": f"Using first product (Item {first_product.get('Item Number', '?')})"
                    })
        
        # Shipment number comparison (2-way: Invoice vs BL) - common for both scenarios
        invoice_shipment = invoice_data.get("shipment_number", "")
        bl_shipment = bl_data.get("shipment_number", "")
        
        if invoice_shipment and bl_shipment:
            total_comparisons += 1
            is_match, similarity = self.fuzzy_match(invoice_shipment, bl_shipment)
            
            if is_match:
                matches += 1
                comparison_result["matching_fields"].append({
                    "field": "shipment_number / shipment_number",
                    "invoice_value": invoice_shipment,
                    "bl_value": bl_shipment,
                    "similarity": round(similarity, 3)
                })
            elif similarity > 0.3:
                comparison_result["discrepancies"].append({
                    "field": "shipment_number / shipment_number",
                    "invoice_value": invoice_shipment,
                    "bl_value": bl_shipment,
                    "similarity": round(similarity, 3)
                })
        
        comparison_result["total_fields_compared"] = total_comparisons
        
        if total_comparisons > 0:
            comparison_result["confidence_score"] = round(matches / total_comparisons, 3)
            comparison_result["documents_related"] = comparison_result["confidence_score"] > 0.5
        
        print(f"\nComparison complete: {matches}/{total_comparisons} fields matched")
        print(f"Confidence score: {comparison_result['confidence_score']:.1%}")
        
        return comparison_result

    def validate_product_info(self, formd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ALL products from Form D against product database"""
        product_validation = {
            "found_matches": False,
            "product_validations": [],
            "validation_process": "Match HS codes first, then validate Form D description"
        }
        
        if self.product_data is None:
            product_validation["error"] = "Product data not available"
            return product_validation
        
        formd_products = formd_data.get("products", [])
        
        if not formd_products:
            product_validation["error"] = "No products found in Form D"
            return product_validation
        
        all_products_valid = True
        
        for product in formd_products:
            formd_description = str(product.get("Product Description", "")).strip()
            formd_hs_code = str(product.get("HS CODE", "")).strip()
            item_number = product.get("Item Number", "")
            
            product_result = {
                "item_number": item_number,
                "formd_hs_code": formd_hs_code,
                "formd_description": formd_description,
                "matches": [],
                "found_match": False
            }
            
            if not formd_hs_code or formd_hs_code == "Not found":
                product_result["error"] = "HS CODE not found"
                all_products_valid = False
                product_validation["product_validations"].append(product_result)
                continue
            
            if not formd_description or formd_description == "Not found":
                product_result["error"] = "Product description not found"
                all_products_valid = False
                product_validation["product_validations"].append(product_result)
                continue
            
            normalized_formd_desc = formd_description.upper().replace(" ", "")
            best_matches = []
            
            for _, row in self.product_data.iterrows():
                hs_code_from_csv = str(row.get("H.S code", "")).strip()
                
                if hs_code_from_csv and hs_code_from_csv != "Not found":
                    hs_code_match = False
                    hs_code_similarity = 0.0
                    
                    if formd_hs_code == hs_code_from_csv:
                        hs_code_match = True
                        hs_code_similarity = 1.0
                    else:
                        _, hs_code_similarity = self.fuzzy_match(formd_hs_code, hs_code_from_csv, 0.8)
                        hs_code_match = hs_code_similarity >= 0.8
                    
                    if hs_code_match:
                        form_d_desc_from_csv = str(row.get("Form D description", "")).strip()
                        
                        if form_d_desc_from_csv and form_d_desc_from_csv != "Not found":
                            normalized_csv_desc = form_d_desc_from_csv.upper().replace(" ", "")
                            description_exists = normalized_csv_desc in normalized_formd_desc
                            _, description_similarity = self.fuzzy_match_description(formd_description, form_d_desc_from_csv)
                            
                            reverse_match = False
                            if len(normalized_csv_desc) > 20:
                                words_in_csv = [word for word in normalized_csv_desc.split() if len(word) > 3]
                                if words_in_csv:
                                    matches_found = sum(1 for word in words_in_csv if word in normalized_formd_desc)
                                    reverse_match = (matches_found / len(words_in_csv)) >= 0.7
                            
                            overall_match = description_exists or reverse_match or description_similarity >= 0.8
                            
                            best_matches.append({
                                "csv_hs_code": hs_code_from_csv,
                                "hs_code_similarity": round(hs_code_similarity, 3),
                                "csv_form_d_description": form_d_desc_from_csv,
                                "description_similarity": round(description_similarity, 3),
                                "sap_code": row.get("SAP code", "Not found"),
                                "overall_match": overall_match
                            })
            
            best_matches.sort(key=lambda x: (x["hs_code_similarity"], x["description_similarity"]), reverse=True)
            
            # *** MODIFICATION: Keep only top 3 matches ***
            product_result["matches"] = best_matches[:3]
            product_result["total_matches_found"] = len(best_matches)  # Track total before truncation
            product_result["found_match"] = any(match["overall_match"] for match in best_matches[:3])
            
            if not product_result["found_match"]:
                all_products_valid = False
            
            product_validation["product_validations"].append(product_result)
        
        product_validation["found_matches"] = all_products_valid
        product_validation["total_products"] = len(formd_products)
        product_validation["valid_products"] = sum(1 for p in product_validation["product_validations"] if p.get("found_match"))
        
        return product_validation
    
    def validate_food_supplement(self, bl_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate if 'FOOD SUPPLEMENT' is in BL description"""
        food_supplement_validation = {
            "contains_food_supplement": False,
            "description_of_goods": "",
            "normalized_description": "",
            "match_details": {}
        }
        
        description = str(bl_data.get("description_of_goods", "")).strip()
        
        if not description or description == "Not found":
            food_supplement_validation["error"] = "Description of goods not found in BL"
            return food_supplement_validation
        
        food_supplement_validation["description_of_goods"] = description
        
        normalized_desc = description.upper().replace(" ", "").replace("-", "").replace("_", "")
        food_supplement_validation["normalized_description"] = normalized_desc
        
        search_terms = ["FOODSUPPLEMENT", "FOODSUPPLEMENTS", "DIETARYSUPPLEMENT"]
        
        for term in search_terms:
            if term in normalized_desc:
                food_supplement_validation["contains_food_supplement"] = True
                food_supplement_validation["match_details"]["matched_term"] = term
                food_supplement_validation["match_details"]["original_description"] = description
                break
        
        if not food_supplement_validation["contains_food_supplement"]:
            upper_desc = description.upper()
            flexible_terms = ["FOOD SUPPLEMENT", "FOOD SUPPLEMENTS", "DIETARY SUPPLEMENT"]
            
            for term in flexible_terms:
                if term in upper_desc:
                    food_supplement_validation["contains_food_supplement"] = True
                    food_supplement_validation["match_details"]["matched_term"] = term
                    food_supplement_validation["match_details"]["original_description"] = description
                    break
        
        return food_supplement_validation

    def validate_company_info(self, formd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Consignee's address by matching Company name"""
        company_validation = {
            "found_matches": False,
            "matches": [],
            "validation_process": "Match Company name first, then compare combined name+address"
        }
        
        if self.company_data is None:
            company_validation["error"] = "Company data not available"
            return company_validation
        
        consignee_name = str(formd_data.get("Consignee's business name", "")).strip()
        consignee_address = str(formd_data.get("Consignee's address", "")).strip()
        
        if not consignee_name or consignee_name == "Not found":
            company_validation["error"] = "Consignee's business name not found"
            return company_validation
        
        best_matches = []
        
        for _, row in self.company_data.iterrows():
            company_name_from_csv = str(row.get("Company name", "")).strip()
            if company_name_from_csv and company_name_from_csv != "Not found":
                is_match, similarity = self.fuzzy_match(consignee_name, company_name_from_csv, 0.7)
                
                if is_match:
                    address_from_csv = str(row.get("Address", "")).strip()
                    
                    address_match = False
                    address_similarity = 0.0
                    
                    if consignee_address and consignee_address != "Not found" and address_from_csv:
                        combined_formd_info = f"{consignee_name} {consignee_address}".strip()
                        address_match, address_similarity = self.fuzzy_match(combined_formd_info, address_from_csv, 0.6)
                    
                    best_matches.append({
                        "formd_consignee_name": consignee_name,
                        "csv_company_name": company_name_from_csv,
                        "company_name_similarity": round(similarity, 3),
                        "formd_consignee_address": consignee_address,
                        "combined_formd_info": f"{consignee_name} {consignee_address}".strip() if consignee_address and consignee_address != "Not found" else consignee_name,
                        "csv_address": address_from_csv,
                        "address_match": address_match,
                        "address_similarity": round(address_similarity, 3),
                        "overall_match": address_match and similarity >= 0.7
                    })
        
        best_matches.sort(key=lambda x: x["company_name_similarity"], reverse=True)
        
        company_validation["matches"] = best_matches
        company_validation["found_matches"] = any(match["overall_match"] for match in best_matches)
        
        return company_validation

    def validate_against_csv(self, formd_data: Dict[str, Any], invoice_data: Dict[str, Any], bl_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted data against CSV files"""
        validation_result = {
            "product_validation": self.validate_product_info(formd_data),
            "company_validation": self.validate_company_info(formd_data),
            "food_supplement_validation": self.validate_food_supplement(bl_data)
        }
        
        return validation_result

    def process_documents(self, formd_pdf_path: str, invoice_pdf_path: str, bl_pdf_path: str, 
                    formd_pages: List[int] = None, invoice_page: int = 2, bl_pages: List[int] = None,
                    rotate_invoice: bool = True, crop_bottom_percent: float = 18) -> Dict[str, Any]:
        """Process Form D, invoice, and BL files - supports multiple products and containers"""
        try:
            print("="*60)
            print("EXTRACTING FORM D INFORMATION")
            print("="*60)
            formd_data = self.extract_formd_info(formd_pdf_path, formd_pages)
            print("Form D Data:")
            print(json.dumps(formd_data, indent=2))
            
            print("\n" + "="*60)
            print("EXTRACTING INVOICE INFORMATION")
            print("="*60)
            invoice_data = self.extract_invoice_info(invoice_pdf_path, invoice_page, rotate_invoice, crop_bottom_percent)
            print("Invoice Data:")
            print(json.dumps(invoice_data, indent=2))
            
            print("\n" + "="*60)
            print("EXTRACTING BL (BILL OF LADING) INFORMATION")
            print("="*60)
            bl_data = self.extract_bl_info(bl_pdf_path, bl_pages, crop_bottom_percent)
            print("BL Data:")
            print(json.dumps(bl_data, indent=2))
            
            print("\n" + "="*60)
            print("COMPARING DOCUMENTS")
            print("="*60)
            comparison_result = self.compare_documents(formd_data, invoice_data, bl_data)
            print("Comparison Result:")
            print(json.dumps(comparison_result, indent=2))
            
            print("\n" + "="*60)
            print("VALIDATING AGAINST CSV DATA")
            print("="*60)
            validation_result = self.validate_against_csv(formd_data, invoice_data, bl_data)
            print("Validation Result:")
            print(json.dumps(validation_result, indent=2))
            
            final_result = {
                "formd_data": formd_data,
                "invoice_data": invoice_data,
                "bl_data": bl_data,
                "comparison": comparison_result,
                "validation": validation_result,
                "overall_assessment": self.generate_overall_assessment(comparison_result, validation_result)
            }
            
            return final_result
            
        except Exception as e:
            print(f"Error processing documents: {e}")
            return {"error": str(e)}

    def generate_overall_assessment(self, comparison_result: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall assessment"""
        assessment = {
            "documents_match": comparison_result.get("documents_related", False),
            "confidence_level": "Low",
            "validation_status": "Unknown",
            "recommendations": [],
            "summary": ""
        }
        
        confidence_score = comparison_result.get("confidence_score", 0)
        
        if confidence_score > 0.8:
            assessment["confidence_level"] = "High"
        elif confidence_score > 0.6:
            assessment["confidence_level"] = "Medium"
        else:
            assessment["confidence_level"] = "Low"
        
        company_valid = validation_result.get("company_validation", {}).get("found_matches", False)
        product_valid = validation_result.get("product_validation", {}).get("found_matches", False)
        food_supplement_valid = validation_result.get("food_supplement_validation", {}).get("contains_food_supplement", False)
        
        validation_count = sum([company_valid, product_valid, food_supplement_valid])
        
        if validation_count >= 2:
            assessment["validation_status"] = "Valid"
        elif validation_count == 1:
            assessment["validation_status"] = "Partially Valid"
        else:
            assessment["validation_status"] = "Invalid"
        
        if not assessment["documents_match"]:
            assessment["recommendations"].append("Documents may not be related - verify manually")
        
        if not company_valid:
            assessment["recommendations"].append("Company information could not be validated against database")
        
        if not product_valid:
            assessment["recommendations"].append("Product information could not be validated against product database")
        
        if not food_supplement_valid:
            assessment["recommendations"].append("BL description does not contain 'FOOD SUPPLEMENT' - verify product type")
        
        if confidence_score < 0.6:
            assessment["recommendations"].append("Low confidence match - manual review recommended")
        
        total_fields = comparison_result.get("total_fields_compared", 0)
        matching_fields = len(comparison_result.get("matching_fields", []))
        
        assessment["summary"] = f"Document comparison: {matching_fields}/{total_fields} fields match " \
                            f"(confidence: {confidence_score:.1%}). Validation: {assessment['validation_status']}"
        
        return assessment

    def print_summary_report(self, result: Dict[str, Any]):
        """Print a concise summary report"""
        print("\n" + "="*60)
        print("SUMMARY REPORT")
        print("="*60)
        
        overall = result.get("overall_assessment", {})
        comparison = result.get("comparison", {})
        validation = result.get("validation", {})
        formd_data = result.get("formd_data", {})
        
        print(f"Document Relationship: {'RELATED' if overall.get('documents_match') else 'NOT RELATED'}")
        print(f"Confidence Level: {overall.get('confidence_level', 'Unknown')}")
        print(f"Validation Status: {overall.get('validation_status', 'Unknown')}")
        
        # Show product count
        total_products = formd_data.get("total_products", 0)
        print(f"\nTotal Products in Form D: {total_products}")
        
        if comparison.get("matching_fields"):
            print(f"\nMatching Common Fields ({len(comparison['matching_fields'])}):")
            for match in comparison["matching_fields"]:
                sim_score = match.get('similarity', 0)
                print(f"  • {match['field']}: {sim_score:.1%} similarity")
        
        # Show product matches
        product_matches = comparison.get("product_matches", [])
        if product_matches:
            print(f"\nProduct Matches ({len(product_matches)}):")
            for match in product_matches:
                print(f"  • Item {match['formd_item']}: {match['similarity']:.1%} similarity {'✓' if match['match'] else '✗'}")
        
        # Validation details
        print(f"\nValidation Results:")
        product_val = validation.get("product_validation", {})
        company_val = validation.get("company_validation", {})
        food_supplement_val = validation.get("food_supplement_validation", {})
        
        print(f"  • Product Validation: {'PASS' if product_val.get('found_matches') else 'FAIL'}")
        if product_val.get("product_validations"):
            valid_count = product_val.get("valid_products", 0)
            total_count = product_val.get("total_products", 0)
            print(f"    - Valid products: {valid_count}/{total_count}")
            for prod_val in product_val["product_validations"]:
                item_num = prod_val.get("item_number", "?")
                found = prod_val.get("found_match", False)
                print(f"      Item {item_num}: {'✓ VALID' if found else '✗ INVALID'}")
        
        print(f"  • Company Validation: {'PASS' if company_val.get('found_matches') else 'FAIL'}")
        if company_val.get("matches"):
            best_company = company_val["matches"][0]
            print(f"    - Best match: {best_company.get('company_name_similarity', 0):.1%} name similarity")
            print(f"    - Address match: {'YES' if best_company.get('address_match') else 'NO'}")
        
        print(f"  • Food Supplement Validation: {'PASS' if food_supplement_val.get('contains_food_supplement') else 'FAIL'}")
        if food_supplement_val.get("match_details"):
            print(f"    - Matched term: {food_supplement_val['match_details'].get('matched_term', 'N/A')}")
        
        if overall.get("recommendations"):
            print(f"\nRecommendations:")
            for rec in overall["recommendations"]:
                print(f"  • {rec}")
        
        print(f"\nSummary: {overall.get('summary', 'No summary available')}")

def main():
    if len(sys.argv) < 4:
        print("Usage: python auditor.py <formd_pdf> <invoice_pdf> <bl_pdf> [options]")
        print("\nOptions:")
        print("  --formd-pages=0,1,2    Comma-separated page numbers for Form D (default: 0)")
        print("  --invoice-page=2       Page number for invoice (default: 2)")
        print("  --bl-pages=0,1         Comma-separated page numbers for BL (default: 0)")
        print("  --api-key=KEY          OpenAI API key")
        print("  --model=MODEL          OpenAI model (default: gpt-4o)")
        print("  --no-rotate            Don't rotate invoice image")
        print("  --crop-bottom=30       Crop percentage from bottom (default: 30)")
        print("\nExamples:")
        print("  python auditor.py formd.pdf invoice.pdf bl.pdf")
        print("  python auditor.py formd.pdf invoice.pdf bl.pdf --formd-pages=0,1,2 --bl-pages=0,1")
        print("  python auditor.py formd.pdf invoice.pdf bl.pdf --formd-pages=0 --invoice-page=2 --bl-pages=0")
        sys.exit(1)
    
    formd_pdf_path = sys.argv[1]
    invoice_pdf_path = sys.argv[2]
    bl_pdf_path = sys.argv[3]
    
    # Default values
    formd_pages = [0]
    invoice_page = 2
    bl_pages = [0]
    api_key = None
    model = "gpt-4o"
    rotate_invoice = True
    crop_bottom_percent = 18.0
    
    # Parse arguments
    for arg in sys.argv[4:]:
        if arg.startswith('--formd-pages='):
            try:
                pages_str = arg.split('=')[1]
                formd_pages = [int(p.strip()) for p in pages_str.split(',')]
                print(f"Form D pages set to: {formd_pages}")
            except ValueError:
                print(f"Warning: Invalid formd-pages format '{arg}'. Using default [0].")
        elif arg.startswith('--invoice-page='):
            try:
                invoice_page = int(arg.split('=')[1])
                print(f"Invoice page set to: {invoice_page}")
            except ValueError:
                print(f"Warning: Invalid invoice-page '{arg}'. Using default 2.")
        elif arg.startswith('--bl-pages='):
            try:
                pages_str = arg.split('=')[1]
                bl_pages = [int(p.strip()) for p in pages_str.split(',')]
                print(f"BL pages set to: {bl_pages}")
            except ValueError:
                print(f"Warning: Invalid bl-pages format '{arg}'. Using default [0].")
        elif arg.startswith('--api-key='):
            api_key = arg.split('=')[1]
        elif arg.startswith('--model='):
            model = arg.split('=')[1]
        elif arg == '--no-rotate':
            rotate_invoice = False
        elif arg.startswith('--crop-bottom='):
            try:
                crop_bottom_percent = float(arg.split('=')[1])
                if crop_bottom_percent < 0 or crop_bottom_percent > 50:
                    print("Warning: crop-bottom percentage should be between 0 and 50. Using default 30%.")
                    crop_bottom_percent = 18.0
            except ValueError:
                print("Warning: Invalid crop-bottom value. Using default 30%.")
                crop_bottom_percent = 18.0
    
    # Validate file paths
    if not os.path.exists(formd_pdf_path):
        print(f"Error: Form D file not found: {formd_pdf_path}")
        sys.exit(1)
    
    if not os.path.exists(invoice_pdf_path):
        print(f"Error: Invoice file not found: {invoice_pdf_path}")
        sys.exit(1)
    
    if not os.path.exists(bl_pdf_path):
        print(f"Error: BL file not found: {bl_pdf_path}")
        sys.exit(1)
    
    if not api_key and not os.getenv('OPENAI_API_KEY'):
        print("Error: OpenAI API key not provided. Please either:")
        print("1. Use --api-key=your_key")
        print("2. Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    try:
        processor = DocumentProcessor(api_key=api_key, model=model)
        
        print(f"\nProcessing settings:")
        print(f"  - Form D pages: {formd_pages}")
        print(f"  - Invoice page: {invoice_page}")
        print(f"  - BL pages: {bl_pages}")
        print(f"  - Rotate invoice: {'Yes' if rotate_invoice else 'No'}")
        print(f"  - Crop bottom: {crop_bottom_percent}%")
        print(f"  - Model: {model}")
        
        result = processor.process_documents(
            formd_pdf_path, 
            invoice_pdf_path,
            bl_pdf_path,
            formd_pages,
            invoice_page,
            bl_pages,
            rotate_invoice, 
            crop_bottom_percent
        )
        
        processor.print_summary_report(result)
        
        base_name = os.path.splitext(os.path.basename(formd_pdf_path))[0]
        invoice_base = os.path.splitext(os.path.basename(invoice_pdf_path))[0]
        bl_base = os.path.splitext(os.path.basename(bl_pdf_path))[0]
        output_file = f"document_analysis_{base_name}_vs_{invoice_base}_vs_{bl_base}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"\nFull results saved to: {output_file}")
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")
            
    except Exception as e:
        print(f"Error initializing processor: {e}")
        if "authentication" in str(e).lower() or "api" in str(e).lower():
            print("Please check your OpenAI API key and ensure it's valid")
        sys.exit(1)

if __name__ == "__main__":
    start_time = time.time()
    
    main()
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n============================================================")
    print(f"Program finished in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"============================================================\n")