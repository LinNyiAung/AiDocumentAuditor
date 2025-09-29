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
        
        # Initialize OpenAI client
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            # Will use OPENAI_API_KEY environment variable
            self.client = OpenAI()
        
        self.model = model
        print(f"Using OpenAI model: {self.model}")
        
        # Load validation data
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
            
        # Try multiple encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        # First try to detect encoding
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
        """Load CSV data for validation with improved encoding handling"""
        try:
            # Load product information
            product_file = './data/5.1 Abbott IR H.S code and FormD desctiption (14-05-24) updated 13Aug 24.csv'
            self.product_data = self.load_csv_with_fallback(product_file)
            
            # Load company information
            company_file = './data/Companies.csv'
            self.company_data = self.load_csv_with_fallback(company_file)
            
            # Load Abbott registration data
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
            # Convert image to base64
            base64_image = self.image_to_base64(image)
            
            # Create the message for OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": question
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            output_text = response.choices[0].message.content
            
            try:
                return json.loads(output_text)
            except json.JSONDecodeError:
                print("Warning: Could not parse JSON response, attempting to clean...")
                # Try to extract JSON from the response
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

    def extract_formd_info(self, formd_pdf_path: str, page_index: int = 0) -> Dict[str, Any]:
        """Extract information from Form D document"""
        try:
            print(f"Converting Form D PDF to image (page {page_index})...")
            images = convert_from_path(formd_pdf_path, dpi=300)
            
            if page_index >= len(images):
                raise ValueError(f"Page {page_index} not found. Form D PDF has {len(images)} pages.")
            
            image = images[page_index]
            
            question = """Retrieve these information from the document: Exporter's business name, Exporter's address, Exporter's country, Consignee's business name, Consignee's address, Consignee's country, Item Number, HS CODE, Product Description, Gross weight, Number of invoices, Date of invoices

Important:
- Do not include the unnecessary information
- Only extract the required information
- Extract information exactly as it appears in the document
- If information is not found, use "Not found" as the value
- Return only valid JSON format
- Do not include any explanations or additional text"""
            
            return self.extract_image_info(image, question)
            
        except Exception as e:
            print(f"Error extracting Form D info: {e}")
            return {"error": str(e)}

    def extract_invoice_info(self, invoice_pdf_path: str, page_index: int = 2, rotate_image: bool = True, crop_bottom_percent: float = 25) -> Dict[str, Any]:
        """Extract information from invoice document with optional rotation and bottom cropping"""
        try:
            print(f"Converting Invoice PDF to image (page {page_index})...")
            images = convert_from_path(invoice_pdf_path, dpi=300)
            
            if page_index >= len(images):
                raise ValueError(f"Page {page_index} not found. Invoice PDF has {len(images)} pages.")
            
            image = images[page_index]
            
            # Rotate image 90 degrees clockwise (to the right) if requested
            if rotate_image:
                print("Rotating invoice image 90 degrees clockwise...")
                image = image.rotate(-90, expand=True)  # -90 for clockwise rotation
            
            # Crop bottom portion of the image if requested
            if crop_bottom_percent > 0:
                width, height = image.size
                crop_pixels = int(height * crop_bottom_percent / 100)
                
                # Crop from top-left (0,0) to bottom-right, but exclude bottom portion
                cropped_height = height - crop_pixels
                print(f"Cropping {crop_bottom_percent}% ({crop_pixels} pixels) from bottom. New height: {cropped_height}")
                
                # Crop box: (left, top, right, bottom)
                crop_box = (0, 0, width, cropped_height)
                image = image.crop(crop_box)
            
            question = """Retrieve these information from the invoice in the image: sold to name, sold to address, ship to name, ship to address, shipped by name, shipped by address, shipment number, number, invoice date, description, total quantity, price per one, total price, packed in, total gross weight kgs, total net weight kgs, total cubic meters

    Important:
    - Do not include the unnecessary information
    - Only extract the required information
    - Extract information exactly as it appears in the document
    - If information is not found, use "Not found" as the value
    - Return only valid JSON format
    - Do not include any explanations or additional text"""
            
            return self.extract_image_info(image, question)
            
        except Exception as e:
            print(f"Error extracting invoice info: {e}")
            return {"error": str(e)}

    def normalize_weight(self, weight_text: str) -> str:
        """Normalize weight by removing spaces, words and letters, leaving only numbers"""
        if not weight_text or weight_text == "Not found" or weight_text == "":
            return ""
        
        # Remove all non-numeric characters (letters, spaces, units, etc.)
        normalized = re.sub(r'[^0-9.]', '', str(weight_text))
        return normalized.strip()

    def normalize_description(self, description_text: str) -> str:
        """Normalize description by removing spaces and making all letters capital"""
        if not description_text or description_text == "Not found" or description_text == "":
            return ""
        
        # Remove all spaces and convert to uppercase
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
        """Perform fuzzy matching between two weight values using normalized weights"""
        if not weight1 or not weight2:
            return False, 0.0
        
        norm_weight1 = self.normalize_weight(weight1)
        norm_weight2 = self.normalize_weight(weight2)
        
        if not norm_weight1 or not norm_weight2:
            return False, 0.0
        
        # For numerical comparison, check if they're exactly the same
        try:
            val1 = float(norm_weight1)
            val2 = float(norm_weight2)
            if val1 == val2:
                return True, 1.0
            # Allow for small numerical differences
            diff = abs(val1 - val2) / max(val1, val2) if max(val1, val2) != 0 else 0
            similarity = 1.0 - diff
            return similarity >= threshold, similarity
        except ValueError:
            # Fall back to string comparison if not numeric
            similarity = difflib.SequenceMatcher(None, norm_weight1, norm_weight2).ratio()
            return similarity >= threshold, similarity

    def fuzzy_match_description(self, desc1: str, desc2: str, threshold: float = 0.8) -> Tuple[bool, float]:
        """Perform fuzzy matching between two descriptions using normalized descriptions"""
        if not desc1 or not desc2:
            return False, 0.0
        
        norm_desc1 = self.normalize_description(desc1)
        norm_desc2 = self.normalize_description(desc2)
        
        if not norm_desc1 or not norm_desc2:
            return False, 0.0
        
        similarity = difflib.SequenceMatcher(None, norm_desc1, norm_desc2).ratio()
        return similarity >= threshold, similarity

    def compare_documents(self, formd_data: Dict[str, Any], invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare Form D and invoice data to determine if they are related"""
        comparison_result = {
            "documents_related": False,
            "matching_fields": [],
            "discrepancies": [],
            "confidence_score": 0.0,
            "total_fields_compared": 0
        }
        
        # Define field mappings between Form D and invoice with specific comparison types
        field_mappings = [
            ("Exporter's business name", "shipped_by_name", "general"),
            ("Consignee's business name", "ship_to_name", "general"),  
            ("Consignee's address", "ship_to_address", "general"),  
            ("Gross weight", "total_gross_weight_kgs", "weight"),
            ("Number of invoices", "number", "general"),
            ("Date of invoices", "invoice_date", "general"),
            ("Product Description", "description", "description")
        ]
        
        matches = 0
        total_comparisons = 0
        
        for formd_field, invoice_field, comparison_type in field_mappings:
            if formd_field in formd_data and invoice_field in invoice_data:
                total_comparisons += 1
                formd_val = formd_data[formd_field]
                invoice_val = invoice_data[invoice_field]
                
                # Use appropriate comparison method based on field type
                if comparison_type == "weight":
                    is_match, similarity = self.fuzzy_match_weight(str(formd_val), str(invoice_val))
                elif comparison_type == "description":
                    is_match, similarity = self.fuzzy_match_description(str(formd_val), str(invoice_val))
                else:
                    is_match, similarity = self.fuzzy_match(str(formd_val), str(invoice_val))
                
                if is_match:
                    matches += 1
                    comparison_result["matching_fields"].append({
                        "field": f"{formd_field} / {invoice_field}",
                        "formd_value": formd_val,
                        "invoice_value": invoice_val,
                        "similarity": round(similarity, 3)
                    })
                elif similarity > 0.3:  # Still record partial matches
                    comparison_result["discrepancies"].append({
                        "field": f"{formd_field} / {invoice_field}",
                        "formd_value": formd_val,
                        "invoice_value": invoice_val,
                        "similarity": round(similarity, 3)
                    })
        
        comparison_result["total_fields_compared"] = total_comparisons
        
        if total_comparisons > 0:
            comparison_result["confidence_score"] = round(matches / total_comparisons, 3)
            comparison_result["documents_related"] = comparison_result["confidence_score"] > 0.5
        
        return comparison_result

    def validate_product_info(self, formd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HS CODE first, then check if Form D description exists in extracted Product Description"""
        product_validation = {
            "found_matches": False,
            "matches": [],
            "validation_process": "Match HS codes first, then validate Form D description existence"
        }
        
        if self.product_data is None:
            product_validation["error"] = "Product data not available"
            return product_validation
        
        formd_description = str(formd_data.get("Product Description", "")).strip()
        formd_hs_code = str(formd_data.get("HS CODE", "")).strip()
        
        if not formd_hs_code or formd_hs_code == "Not found":
            product_validation["error"] = "Form D HS CODE not found"
            return product_validation
        
        if not formd_description or formd_description == "Not found":
            product_validation["error"] = "Form D description not found"
            return product_validation
        
        # Normalize the Form D description for better matching
        normalized_formd_desc = formd_description.upper().replace(" ", "")
        
        best_matches = []
        
        # First, find matching HS codes in product file
        for _, row in self.product_data.iterrows():
            hs_code_from_csv = str(row.get("H.S code", "")).strip()
            
            if hs_code_from_csv and hs_code_from_csv != "Not found":
                # Check if HS codes match (exact or fuzzy)
                hs_code_match = False
                hs_code_similarity = 0.0
                
                if formd_hs_code == hs_code_from_csv:
                    hs_code_match = True
                    hs_code_similarity = 1.0
                else:
                    # Check partial match for HS codes
                    _, hs_code_similarity = self.fuzzy_match(formd_hs_code, hs_code_from_csv, 0.8)
                    hs_code_match = hs_code_similarity >= 0.8
                
                # If HS code matches, then check if Form D description from CSV exists in extracted description
                if hs_code_match:
                    form_d_desc_from_csv = str(row.get("Form D description", "")).strip()
                    
                    if form_d_desc_from_csv and form_d_desc_from_csv != "Not found":
                        # Normalize CSV Form D description for comparison
                        normalized_csv_desc = form_d_desc_from_csv.upper().replace(" ", "")
                        
                        # Check if CSV Form D description exists within the extracted Form D description
                        description_exists = normalized_csv_desc in normalized_formd_desc
                        
                        # Calculate similarity score for reference
                        _, description_similarity = self.fuzzy_match_description(formd_description, form_d_desc_from_csv)
                        
                        # Also check reverse - if extracted description exists in CSV description (partial match)
                        reverse_match = False
                        if len(normalized_csv_desc) > 20:  # Only for reasonably long descriptions
                            # Check if significant portion of CSV description is in Form D description
                            words_in_csv = [word for word in normalized_csv_desc.split() if len(word) > 3]
                            if words_in_csv:
                                matches_found = sum(1 for word in words_in_csv if word in normalized_formd_desc)
                                reverse_match = (matches_found / len(words_in_csv)) >= 0.7
                        
                        overall_match = description_exists or reverse_match or description_similarity >= 0.8
                        
                        best_matches.append({
                            "formd_hs_code": formd_hs_code,
                            "csv_hs_code": hs_code_from_csv,
                            "hs_code_match": hs_code_match,
                            "hs_code_similarity": round(hs_code_similarity, 3),
                            "formd_description": formd_description,
                            "csv_form_d_description": form_d_desc_from_csv,
                            "description_exists_in_formd": description_exists,
                            "reverse_partial_match": reverse_match,
                            "description_similarity": round(description_similarity, 3),
                            "sap_code": row.get("SAP code", "Not found"),
                            "overall_match": overall_match,
                            "match_reason": self._get_match_reason(description_exists, reverse_match, description_similarity)
                        })
        
        # Sort by HS code similarity first, then by description match quality
        best_matches.sort(key=lambda x: (x["hs_code_similarity"], x["description_similarity"]), reverse=True)
        
        product_validation["matches"] = best_matches
        product_validation["found_matches"] = any(match["overall_match"] for match in best_matches)
        
        return product_validation

    def _get_match_reason(self, description_exists: bool, reverse_match: bool, similarity: float) -> str:
        """Get the reason for the match result"""
        if description_exists:
            return "CSV description found in Form D description"
        elif reverse_match:
            return "Significant overlap between descriptions"
        elif similarity >= 0.8:
            return "High similarity score"
        else:
            return "No significant match found"

    def validate_company_info(self, formd_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Consignee's address by first matching Company name"""
        company_validation = {
            "found_matches": False,
            "matches": [],
            "validation_process": "Match Company name first, then compare combined name+address with CSV address"
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
        
        # Find best matching company names
        for _, row in self.company_data.iterrows():
            company_name_from_csv = str(row.get("Company name", "")).strip()
            if company_name_from_csv and company_name_from_csv != "Not found":
                is_match, similarity = self.fuzzy_match(consignee_name, company_name_from_csv, 0.7)
                
                if is_match:
                    address_from_csv = str(row.get("Address", "")).strip()
                    
                    # Check if addresses match - now combining name + address from Form D
                    address_match = False
                    address_similarity = 0.0
                    
                    if consignee_address and consignee_address != "Not found" and address_from_csv:
                        # Combine Form D consignee name and address for comparison
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
        
        # Sort by company name similarity
        best_matches.sort(key=lambda x: x["company_name_similarity"], reverse=True)
        
        company_validation["matches"] = best_matches
        company_validation["found_matches"] = any(match["overall_match"] for match in best_matches)
        
        return company_validation


    def validate_against_csv(self, formd_data: Dict[str, Any], invoice_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate extracted data against CSV files with improved logic"""
        validation_result = {
            "product_validation": self.validate_product_info(formd_data),
            "company_validation": self.validate_company_info(formd_data),
            
        }
        
        return validation_result

    def process_documents(self, formd_pdf_path: str, invoice_pdf_path: str, formd_page: int = 0, invoice_page: int = 2, rotate_invoice: bool = True, crop_bottom_percent: float = 25) -> Dict[str, Any]:
        """Process both Form D and invoice files, compare them, and validate against CSV data"""
        try:
            print("="*60)
            print("EXTRACTING FORM D INFORMATION")
            print("="*60)
            formd_data = self.extract_formd_info(formd_pdf_path, formd_page)
            print("Form D Data:")
            print(json.dumps(formd_data, indent=2))
            
            print("\n" + "="*60)
            print("EXTRACTING INVOICE INFORMATION")
            print("="*60)
            invoice_data = self.extract_invoice_info(invoice_pdf_path, invoice_page, rotate_invoice, crop_bottom_percent)
            print("Invoice Data:")
            print(json.dumps(invoice_data, indent=2))
            
            print("\n" + "="*60)
            print("COMPARING DOCUMENTS")
            print("="*60)
            comparison_result = self.compare_documents(formd_data, invoice_data)
            print("Comparison Result:")
            print(json.dumps(comparison_result, indent=2))
            
            print("\n" + "="*60)
            print("VALIDATING AGAINST CSV DATA")
            print("="*60)
            validation_result = self.validate_against_csv(formd_data, invoice_data)
            print("Validation Result:")
            print(json.dumps(validation_result, indent=2))
            
            # Compile final result
            final_result = {
                "formd_data": formd_data,
                "invoice_data": invoice_data,
                "comparison": comparison_result,
                "validation": validation_result,
                "overall_assessment": self.generate_overall_assessment(comparison_result, validation_result)
            }
            
            return final_result
            
        except Exception as e:
            print(f"Error processing documents: {e}")
            return {"error": str(e)}

    def generate_overall_assessment(self, comparison_result: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an overall assessment of the document relationship and validity"""
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
        
        # Check validation results
        company_valid = validation_result.get("company_validation", {}).get("found_matches", False)
        product_valid = validation_result.get("product_validation", {}).get("found_matches", False)
        
        
        validation_count = sum([company_valid, product_valid])
        
        if validation_count >= 2:
            assessment["validation_status"] = "Valid"
        elif validation_count == 1:
            assessment["validation_status"] = "Partially Valid"
        else:
            assessment["validation_status"] = "Invalid"
        
        # Generate recommendations
        if not assessment["documents_match"]:
            assessment["recommendations"].append("Documents may not be related - verify manually")
        
        if not company_valid:
            assessment["recommendations"].append("Company information could not be validated against database")
        
        if not product_valid:
            assessment["recommendations"].append("Product information could not be validated against product database")
        
        
        if confidence_score < 0.6:
            assessment["recommendations"].append("Low confidence match - manual review recommended")
        
        # Generate summary
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
        
        print(f"Document Relationship: {'RELATED' if overall.get('documents_match') else 'NOT RELATED'}")
        print(f"Confidence Level: {overall.get('confidence_level', 'Unknown')}")
        print(f"Validation Status: {overall.get('validation_status', 'Unknown')}")
        
        if comparison.get("matching_fields"):
            print(f"\nMatching Fields ({len(comparison['matching_fields'])}):")
            for match in comparison["matching_fields"]:
                print(f"  â€¢ {match['field']}: {match['similarity']:.1%} similarity")
        
        # Print validation details
        print(f"\nValidation Results:")
        product_val = validation.get("product_validation", {})
        company_val = validation.get("company_validation", {})
        
        
        print(f"  â€¢ Product Validation: {'PASS' if product_val.get('found_matches') else 'FAIL'}")
        if product_val.get("matches"):
            best_product = product_val["matches"][0]
            print(f"    - Best match: {best_product.get('description_similarity', 0):.1%} description similarity")
            print(f"    - HS Code match: {'YES' if best_product.get('hs_code_match') else 'NO'}")
        
        print(f"  â€¢ Company Validation: {'PASS' if company_val.get('found_matches') else 'FAIL'}")
        if company_val.get("matches"):
            best_company = company_val["matches"][0]
            print(f"    - Best match: {best_company.get('company_name_similarity', 0):.1%} name similarity")
            print(f"    - Address match: {'YES' if best_company.get('address_match') else 'NO'}")
        
        
        if overall.get("recommendations"):
            print(f"\nRecommendations:")
            for rec in overall["recommendations"]:
                print(f"  â€¢ {rec}")
        
        print(f"\nSummary: {overall.get('summary', 'No summary available')}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python document_processor.py <formd_pdf_path> <invoice_pdf_path> [formd_page] [invoice_page] [api_key] [model] [--no-rotate] [--crop-bottom=X]")
        print("Example: python document_processor.py formd.pdf invoice.pdf 0 0")
        print("Example with API key: python document_processor.py formd.pdf invoice.pdf 0 0 your_api_key")
        print("Example with custom model: python document_processor.py formd.pdf invoice.pdf 0 0 your_api_key gpt-4o-mini")
        print("Example without rotation: python document_processor.py formd.pdf invoice.pdf 0 0 your_api_key gpt-4o --no-rotate")
        print("Example with bottom cropping: python document_processor.py formd.pdf invoice.pdf 0 0 your_api_key gpt-4o --crop-bottom=15")
        print("\nNote: If no API key is provided, OPENAI_API_KEY environment variable will be used")
        print("Note: By default, invoice images are rotated 90 degrees clockwise. Use --no-rotate to disable.")
        print("Note: By default, 10% of the bottom is cropped. Use --crop-bottom=X to set percentage (0 to disable).")
        sys.exit(1)
    
    formd_pdf_path = sys.argv[1]
    invoice_pdf_path = sys.argv[2]
    
    # Initialize default values
    formd_page = 0
    invoice_page = 2
    api_key = None
    model = "gpt-4o"
    rotate_invoice = True
    crop_bottom_percent = 25.0  # Default 10% crop
    
    # Parse arguments
    arg_index = 3
    while arg_index < len(sys.argv):
        arg = sys.argv[arg_index]
        
        if arg == '--no-rotate':
            rotate_invoice = False
        elif arg.startswith('--crop-bottom='):
            try:
                crop_bottom_percent = float(arg.split('=')[1])
                if crop_bottom_percent < 0 or crop_bottom_percent > 50:
                    print("Warning: crop-bottom percentage should be between 0 and 50. Using default 10%.")
                    crop_bottom_percent = 25.0
            except ValueError:
                print("Warning: Invalid crop-bottom value. Using default 10%.")
                crop_bottom_percent = 25.0
        elif arg_index == 3:  # formd_page
            try:
                formd_page = int(arg)
            except ValueError:
                print(f"Warning: Invalid formd_page '{arg}'. Using default 0.")
        elif arg_index == 4:  # invoice_page
            try:
                invoice_page = int(arg)
            except ValueError:
                print(f"Warning: Invalid invoice_page '{arg}'. Using default 0.")
        elif arg_index == 5:  # api_key
            api_key = arg
        elif arg_index == 6:  # model
            model = arg
        
        arg_index += 1
    
    # Validate file paths
    if not os.path.exists(formd_pdf_path):
        print(f"Error: Form D file not found: {formd_pdf_path}")
        sys.exit(1)
    
    if not os.path.exists(invoice_pdf_path):
        print(f"Error: Invoice file not found: {invoice_pdf_path}")
        sys.exit(1)
    
    # Check if API key is available
    if not api_key and not os.getenv('OPENAI_API_KEY'):
        print("Error: OpenAI API key not provided. Please either:")
        print("1. Pass API key as argument: python document_processor.py ... your_api_key")
        print("2. Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    try:
        processor = DocumentProcessor(api_key=api_key, model=model)
        
        print(f"Processing settings:")
        print(f"  - Form D page: {formd_page}")
        print(f"  - Invoice page: {invoice_page}")
        print(f"  - Rotate invoice: {'Yes' if rotate_invoice else 'No'}")
        print(f"  - Crop bottom: {crop_bottom_percent}%")
        print(f"  - Model: {model}")
        
        result = processor.process_documents(
            formd_pdf_path, 
            invoice_pdf_path, 
            formd_page, 
            invoice_page, 
            rotate_invoice, 
            crop_bottom_percent
        )
        
        # Print summary report
        processor.print_summary_report(result)
        
        # Save results to file
        base_name = os.path.splitext(os.path.basename(formd_pdf_path))[0]
        invoice_base = os.path.splitext(os.path.basename(invoice_pdf_path))[0]
        output_file = f"document_analysis_{base_name}_vs_{invoice_base}.json"
        
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
    
    main()   # run your program
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\n============================================================")
    print(f"Program finished in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"============================================================\n")