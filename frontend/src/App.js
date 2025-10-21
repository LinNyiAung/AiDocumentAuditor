import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, XCircle, AlertCircle, Loader2, Download, ChevronDown, ChevronRight, Settings } from 'lucide-react';

const DocumentProcessor = () => {
  const [formDFile, setFormDFile] = useState(null);
  const [invoiceFile, setInvoiceFile] = useState(null);
  const [blFile, setBlFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expandedSections, setExpandedSections] = useState({});
  const [showAdvanced, setShowAdvanced] = useState(false);
  
  // Advanced settings
  const [formDPages, setFormDPages] = useState('0');
  const [invoicePage, setInvoicePage] = useState('2');
  const [blPages, setBlPages] = useState('0');

  const toggleSection = (sectionId) => {
    setExpandedSections(prev => ({
      ...prev,
      [sectionId]: !prev[sectionId]
    }));
  };

  const handleFileChange = (e, fileType) => {
    const file = e.target.files[0];
    if (file && file.type === 'application/pdf') {
      if (fileType === 'formD') {
        setFormDFile(file);
      } else if (fileType === 'invoice') {
        setInvoiceFile(file);
      } else if (fileType === 'bl') {
        setBlFile(file);
      }
      setError(null);
    } else {
      setError('Please select a valid PDF file');
    }
  };

  const handleSubmit = async () => {
    if (!formDFile || !invoiceFile || !blFile) {
      setError('Please select all three PDF files: Form D, Invoice, and BL');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('formd_pdf', formDFile);
    formData.append('invoice_pdf', invoiceFile);
    formData.append('bl_pdf', blFile);
    formData.append('formd_pages', formDPages);
    formData.append('invoice_page', invoicePage);
    formData.append('bl_pages', blPages);

    try {
      const response = await fetch('http://localhost:8000/process-documents', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(`Processing failed: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadResults = () => {
    if (result) {
      const dataStr = JSON.stringify(result.data, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `document_analysis_${Date.now()}.json`;
      link.click();
      URL.revokeObjectURL(url);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'PASS':
      case 'Valid':
      case 'RELATED':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'FAIL':
      case 'Invalid':
      case 'NOT RELATED':
        return <XCircle className="w-5 h-5 text-red-500" />;
      case 'Partially Valid':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'PASS':
      case 'Valid':
      case 'RELATED':
        return 'text-green-700 bg-green-100';
      case 'FAIL':
      case 'Invalid':
      case 'NOT RELATED':
        return 'text-red-700 bg-red-100';
      case 'Partially Valid':
        return 'text-yellow-700 bg-yellow-100';
      default:
        return 'text-gray-700 bg-gray-100';
    }
  };

  
  const highlightMatchingText = (formdDesc, csvDesc) => {
    if (!formdDesc || !csvDesc) return formdDesc;
    
    // Normalize: remove spaces and convert to uppercase
    const normalizeForComparison = (str) => {
      return str.toUpperCase().replace(/\s+/g, '');
    };
    
    const normalizedFormD = normalizeForComparison(formdDesc);
    const normalizedCSV = normalizeForComparison(csvDesc);
    
    // Find where CSV description appears in Form D description
    const matchIndex = normalizedFormD.indexOf(normalizedCSV);
    
    if (matchIndex !== -1) {
      // Found exact match - map back to original positions
      const startIdx = matchIndex;
      const endIdx = startIdx + normalizedCSV.length;
      
      // Map normalized positions back to original string positions
      let charCount = 0;
      let startPos = 0;
      let endPos = formdDesc.length;
      let foundStart = false;
      let foundEnd = false;
      
      for (let i = 0; i < formdDesc.length; i++) {
        const char = formdDesc[i];
        
        // Count non-space characters
        if (!/\s/.test(char)) {
          if (charCount === startIdx && !foundStart) {
            startPos = i;
            foundStart = true;
          }
          if (charCount === endIdx - 1 && !foundEnd) {
            // Find the end of the current word (including punctuation)
            endPos = i + 1;
            // Continue to the end of the word
            while (endPos < formdDesc.length && !/\s/.test(formdDesc[endPos])) {
              endPos++;
            }
            foundEnd = true;
            break;
          }
          charCount++;
        }
      }
      
      if (foundStart && foundEnd) {
        return (
          <>
            {formdDesc.substring(0, startPos)}
            <mark className="bg-yellow-200 font-medium">{formdDesc.substring(startPos, endPos)}</mark>
            {formdDesc.substring(endPos)}
          </>
        );
      }
    }
    
    // No exact match found - return original text without highlighting
    return formdDesc;
  };

  const ProductValidationCard = ({ validation, index }) => {
    // Get the best match (highest similarity) regardless of overall_match status
    const bestMatch = validation.matches && validation.matches.length > 0 ? validation.matches[0] : null;
    const hasCompleteMatch = bestMatch?.overall_match || false;
    
    return (
      <div className="border rounded-lg p-4 bg-gray-50">
        <div className="flex justify-between items-start mb-3">
          <div className="flex items-center">
            {getStatusIcon(validation.found_match ? 'PASS' : 'FAIL')}
            <h5 className="ml-2 font-medium text-gray-900">
              Item {validation.item_number || index + 1}
            </h5>
          </div>
          <span className={`text-xs px-2 py-1 rounded ${validation.found_match ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
            {validation.found_match ? 'Valid' : 'Invalid'}
          </span>
        </div>

        <div className="space-y-2 mb-3">
          <div>
            <span className="text-xs font-medium text-gray-500">HS Code:</span>
            <div className="text-sm font-mono bg-white p-2 rounded border mt-1">
              {validation.formd_hs_code || 'Not found'}
            </div>
          </div>
          <div>
            <span className="text-xs font-medium text-gray-500">Description:</span>
            <div className="text-sm bg-white p-2 rounded border mt-1">
              {bestMatch ? (
                highlightMatchingText(validation.formd_description, bestMatch.csv_form_d_description)
              ) : (
                validation.formd_description || 'Not found'
              )}
            </div>
          </div>
        </div>

        {bestMatch ? (
          <div className="space-y-2">
            <span className="text-xs font-medium text-gray-500">
              {hasCompleteMatch ? 'Matched Product:' : 'Most Similar Product:'}
            </span>
            <div className={`p-2 rounded border ${hasCompleteMatch ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'}`}>
              <div className="flex justify-between items-start mb-1">
                <span className={`text-xs font-medium ${hasCompleteMatch ? 'text-green-800' : 'text-yellow-800'}`}>
                  {hasCompleteMatch ? 'Complete Match' : 'Partial Match'}
                </span>
                <div className="flex gap-1">
                  <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                    HS: {(bestMatch.hs_code_similarity * 100).toFixed(0)}%
                  </span>
                  <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded">
                    Desc: {(bestMatch.description_similarity * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="text-xs text-gray-600 space-y-1">
                <div><span className="font-medium">CSV HS:</span> {bestMatch.csv_hs_code}</div>
                <div><span className="font-medium">CSV Desc:</span> {bestMatch.csv_form_d_description}</div>
                {bestMatch.sap_code && bestMatch.sap_code !== 'Not found' && (
                  <div><span className="font-medium">SAP:</span> {bestMatch.sap_code}</div>
                )}
              </div>
              {!hasCompleteMatch && (
                <div className="mt-2 text-xs text-yellow-700 italic">
                  Note: Description does not exactly match. Manual verification recommended.
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="p-2 bg-red-50 border border-red-200 rounded text-xs text-red-800">
            No matches found in database
          </div>
        )}
      </div>
    );
  };

  const ThreeWayComparisonCard = ({ field }) => (
    <div className="border rounded-lg p-4 bg-gray-50">
      <div className="flex justify-between items-start mb-3">
        <h5 className="font-medium text-gray-900">{field.field}</h5>
        <span className="text-xs text-blue-700 bg-blue-100 px-2 py-1 rounded">
          Avg: {((field.avg_similarity || field.similarity) * 100).toFixed(1)}%
        </span>
      </div>
      
      <div className="space-y-3">
        {field.formd_value !== undefined && (
          <div>
            <span className="text-xs font-medium text-gray-500 uppercase">Form D:</span>
            <div className="mt-1 p-2 bg-white rounded border text-sm">
              {field.formd_value || 'Not found'}
            </div>
          </div>
        )}
        
        {field.invoice_value !== undefined && (
          <div>
            <span className="text-xs font-medium text-gray-500 uppercase">Invoice:</span>
            <div className="mt-1 p-2 bg-white rounded border text-sm">
              {field.invoice_value || 'Not found'}
            </div>
          </div>
        )}
        
        {field.bl_value !== undefined && (
          <div>
            <span className="text-xs font-medium text-gray-500 uppercase">BL:</span>
            <div className="mt-1 p-2 bg-white rounded border text-sm">
              {field.bl_value || 'Not found'}
            </div>
          </div>
        )}
      </div>
      
      {field.formd_invoice_sim !== undefined && (
        <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
          <div className="text-center p-1 bg-blue-50 rounded">
            <div className="text-gray-500">F↔I</div>
            <div className="font-medium">{(field.formd_invoice_sim * 100).toFixed(0)}%</div>
          </div>
          <div className="text-center p-1 bg-blue-50 rounded">
            <div className="text-gray-500">F↔B</div>
            <div className="font-medium">{(field.formd_bl_sim * 100).toFixed(0)}%</div>
          </div>
          <div className="text-center p-1 bg-blue-50 rounded">
            <div className="text-gray-500">I↔B</div>
            <div className="font-medium">{(field.invoice_bl_sim * 100).toFixed(0)}%</div>
          </div>
        </div>
      )}
      
      {field.note && (
        <div className="mt-2 text-xs text-gray-500 italic">{field.note}</div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Document Processor</h1>
          <p className="text-lg text-gray-600">Upload Form D, Invoice, and BL PDFs for AI-powered analysis</p>
          <p className="text-sm text-gray-500 mt-1">Supports multiple pages and products</p>
        </div>

        <div className="grid lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <Upload className="w-5 h-5 mr-2" />
                Upload Documents
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Form D PDF
                  </label>
                  <div className="relative">
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={(e) => handleFileChange(e, 'formD')}
                      className="hidden"
                      id="formD"
                    />
                    <label
                      htmlFor="formD"
                      className="flex flex-col items-center justify-center w-full h-28 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer hover:bg-gray-50 hover:border-gray-400 transition-colors"
                    >
                      <div className="flex flex-col items-center justify-center pt-4 pb-5">
                        <FileText className="w-7 h-7 text-gray-400 mb-2" />
                        <p className="text-sm text-gray-500 text-center px-2">
                          {formDFile ? formDFile.name : 'Click to upload Form D'}
                        </p>
                      </div>
                    </label>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Invoice PDF
                  </label>
                  <div className="relative">
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={(e) => handleFileChange(e, 'invoice')}
                      className="hidden"
                      id="invoice"
                    />
                    <label
                      htmlFor="invoice"
                      className="flex flex-col items-center justify-center w-full h-28 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer hover:bg-gray-50 hover:border-gray-400 transition-colors"
                    >
                      <div className="flex flex-col items-center justify-center pt-4 pb-5">
                        <FileText className="w-7 h-7 text-gray-400 mb-2" />
                        <p className="text-sm text-gray-500 text-center px-2">
                          {invoiceFile ? invoiceFile.name : 'Click to upload Invoice'}
                        </p>
                      </div>
                    </label>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    BL (Bill of Lading) PDF
                  </label>
                  <div className="relative">
                    <input
                      type="file"
                      accept=".pdf"
                      onChange={(e) => handleFileChange(e, 'bl')}
                      className="hidden"
                      id="bl"
                    />
                    <label
                      htmlFor="bl"
                      className="flex flex-col items-center justify-center w-full h-28 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer hover:bg-gray-50 hover:border-gray-400 transition-colors"
                    >
                      <div className="flex flex-col items-center justify-center pt-4 pb-5">
                        <FileText className="w-7 h-7 text-gray-400 mb-2" />
                        <p className="text-sm text-gray-500 text-center px-2">
                          {blFile ? blFile.name : 'Click to upload BL'}
                        </p>
                      </div>
                    </label>
                  </div>
                </div>

                <div className="border-t pt-4">
                  <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center text-sm text-gray-600 hover:text-gray-900 mb-3"
                  >
                    <Settings className="w-4 h-4 mr-2" />
                    Advanced Settings
                    {showAdvanced ? <ChevronDown className="w-4 h-4 ml-1" /> : <ChevronRight className="w-4 h-4 ml-1" />}
                  </button>

                  {showAdvanced && (
                    <div className="space-y-3 bg-gray-50 p-3 rounded-lg">
                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1">
                          Form D Pages (comma-separated)
                        </label>
                        <input
                          type="text"
                          value={formDPages}
                          onChange={(e) => setFormDPages(e.target.value)}
                          placeholder="e.g., 0,1,2"
                          className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                        />
                        <p className="text-xs text-gray-500 mt-1">Default: 0</p>
                      </div>

                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1">
                          Invoice Page
                        </label>
                        <input
                          type="number"
                          value={invoicePage}
                          onChange={(e) => setInvoicePage(e.target.value)}
                          min="0"
                          className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                        />
                        <p className="text-xs text-gray-500 mt-1">Default: 2</p>
                      </div>

                      <div>
                        <label className="block text-xs font-medium text-gray-700 mb-1">
                          BL Pages (comma-separated)
                        </label>
                        <input
                          type="text"
                          value={blPages}
                          onChange={(e) => setBlPages(e.target.value)}
                          placeholder="e.g., 0,1"
                          className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                        />
                        <p className="text-xs text-gray-500 mt-1">Default: 0</p>
                      </div>
                    </div>
                  )}
                </div>

                <button
                  onClick={handleSubmit}
                  disabled={!formDFile || !invoiceFile || !blFile || isProcessing}
                  className="w-full bg-indigo-600 text-white py-3 px-4 rounded-lg font-medium hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    'Process Documents'
                  )}
                </button>
              </div>

              {error && (
                <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-lg text-sm">
                  {error}
                </div>
              )}
            </div>
          </div>

          <div className="lg:col-span-2">
            {result ? (
              <div className="space-y-6">
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-gray-900">Analysis Summary</h2>
                    <button
                      onClick={downloadResults}
                      className="flex items-center px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors text-sm"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </button>
                  </div>

                  {result.data?.overall_assessment && (
                    <div className="space-y-4">
                      <div className="grid md:grid-cols-2 gap-4">
                        <div className="space-y-3">
                          <div className="flex items-center">
                            {getStatusIcon(result.data.overall_assessment.documents_match ? 'RELATED' : 'NOT RELATED')}
                            <span className="ml-2 font-medium">
                              Documents: {result.data.overall_assessment.documents_match ? 'RELATED' : 'NOT RELATED'}
                            </span>
                          </div>
                          
                          <div className={`px-3 py-1 rounded-full text-sm font-medium inline-flex items-center ${getStatusColor(result.data.overall_assessment.validation_status)}`}>
                            {getStatusIcon(result.data.overall_assessment.validation_status)}
                            <span className="ml-2">Validation: {result.data.overall_assessment.validation_status}</span>
                          </div>
                        </div>
                        
                        <div className="space-y-2">
                          <div className="text-sm text-gray-600">
                            Confidence: <span className="font-semibold text-gray-900">{result.data.overall_assessment.confidence_level}</span>
                          </div>
                          <div className="text-sm text-gray-600">
                            Match Score: <span className="font-semibold text-gray-900">{(result.data.comparison?.confidence_score * 100 || 0).toFixed(1)}%</span>
                          </div>
                          {result.data?.formd_data?.total_products && (
                            <div className="text-sm text-gray-600">
                              Total Products: <span className="font-semibold text-gray-900">{result.data.formd_data.total_products}</span>
                            </div>
                          )}
                        </div>
                      </div>

                      {result.metadata && (
                        <div className="text-xs text-gray-500 bg-gray-50 p-3 rounded-lg">
                          <div className="grid grid-cols-2 gap-2">
                            <div>Form D Pages: {result.metadata.formd_pages?.join(', ') || '0'}</div>
                            <div>Invoice Page: {result.metadata.invoice_page}</div>
                            <div>BL Pages: {result.metadata.bl_pages?.join(', ') || '0'}</div>
                          </div>
                        </div>
                      )}

                      {result.data.overall_assessment.summary && (
                        <div className="p-3 bg-gray-50 rounded-lg">
                          <p className="text-sm text-gray-700">{result.data.overall_assessment.summary}</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>

                {result.data?.validation && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-gray-900">Validation Results</h3>
                      <button
                        onClick={() => toggleSection('validation')}
                        className="flex items-center text-sm text-gray-500 hover:text-gray-700"
                      >
                        {expandedSections.validation ? (
                          <>
                            <ChevronDown className="w-4 h-4 mr-1" />
                            Collapse
                          </>
                        ) : (
                          <>
                            <ChevronRight className="w-4 h-4 mr-1" />
                            Expand Details
                          </>
                        )}
                      </button>
                    </div>
                    
                    <div className="space-y-4">
                      {result.data.validation.product_validation && (
                        <div className="border rounded-lg p-4">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center">
                              {getStatusIcon(result.data.validation.product_validation.found_matches ? 'PASS' : 'FAIL')}
                              <h4 className="ml-2 font-medium">Product Validation</h4>
                            </div>
                            <span className="text-xs text-gray-500">
                              {result.data.validation.product_validation.valid_products || 0} / {result.data.validation.product_validation.total_products || 0} valid
                            </span>
                          </div>
                          
                          {expandedSections.validation && (
                            <div className="mt-4 space-y-4">
                              {result.data.validation.product_validation.product_validations?.length > 0 ? (
                                result.data.validation.product_validation.product_validations.map((validation, idx) => (
                                  <ProductValidationCard key={idx} validation={validation} index={idx} />
                                ))
                              ) : (
                                <div className="text-sm text-gray-500 text-center py-4">
                                  No product validation data available
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      )}

                      {result.data.validation.company_validation && (
                        <div className="border rounded-lg p-4">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center">
                              {getStatusIcon(result.data.validation.company_validation.found_matches ? 'PASS' : 'FAIL')}
                              <h4 className="ml-2 font-medium">Company Validation</h4>
                            </div>
                            <span className="text-xs text-gray-500">
                              {result.data.validation.company_validation.matches?.length || 0} matches found
                            </span>
                          </div>
                          
                          {expandedSections.validation && result.data.validation.company_validation.matches?.length > 0 && (
                            <div className="mt-4 space-y-3">
                              {result.data.validation.company_validation.matches.slice(0, 3).map((match, idx) => (
                                <div key={idx} className={`p-3 rounded border ${match.overall_match ? 'bg-green-50 border-green-200' : 'bg-white'}`}>
                                  <div className="flex justify-between items-start mb-2">
                                    <span className="text-sm font-medium">Match {idx + 1}</span>
                                    <div className="flex gap-1">
                                      <span className="text-xs bg-blue-100 text-blue-700 px-2 py-0.5 rounded">
                                        Name: {(match.company_name_similarity * 100).toFixed(0)}%
                                      </span>
                                      {match.address_match && (
                                        <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">
                                          Address ✓
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                  <div className="space-y-2 text-xs">
                                    <div>
                                      <span className="font-medium text-gray-600">Form D:</span>
                                      <div className="text-gray-700 mt-1">{match.formd_consignee_name}</div>
                                    </div>
                                    <div>
                                      <span className="font-medium text-gray-600">CSV:</span>
                                      <div className="text-gray-700 mt-1">{match.csv_company_name}</div>
                                    </div>
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      {result.data.validation.food_supplement_validation && (
                        <div className="border rounded-lg p-4">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center">
                              {getStatusIcon(result.data.validation.food_supplement_validation.contains_food_supplement ? 'PASS' : 'FAIL')}
                              <h4 className="ml-2 font-medium">Food Supplement Validation</h4>
                            </div>
                            <span className={`text-xs px-2 py-1 rounded ${result.data.validation.food_supplement_validation.contains_food_supplement ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                              {result.data.validation.food_supplement_validation.contains_food_supplement ? 'Found' : 'Not Found'}
                            </span>
                          </div>
                          
                          {expandedSections.validation && result.data.validation.food_supplement_validation.match_details?.matched_term && (
                            <div className="mt-3 p-2 bg-green-50 border border-green-200 rounded text-xs">
                              <span className="font-medium">Matched:</span> {result.data.validation.food_supplement_validation.match_details.matched_term}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {result.data?.comparison?.matching_fields?.length > 0 && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-gray-900">Matching Fields</h3>
                      <span className="text-sm text-gray-500">
                        {result.data.comparison.matching_fields.length} fields matched
                      </span>
                    </div>
                    
                    <div className="space-y-4">
                      {result.data.comparison.matching_fields.map((field, idx) => (
                        <ThreeWayComparisonCard key={idx} field={field} />
                      ))}
                    </div>
                  </div>
                )}

                {result.data?.comparison?.discrepancies?.length > 0 && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Discrepancies</h3>
                    <div className="space-y-4">
                      {result.data.comparison.discrepancies.map((field, idx) => (
                        <div key={idx} className="border border-red-200 rounded-lg p-4 bg-red-50">
                          <div className="flex justify-between items-start mb-3">
                            <h5 className="font-medium text-gray-900">{field.field}</h5>
                            <span className="text-sm text-red-700 bg-red-200 px-2 py-1 rounded">
                              {((field.avg_similarity || field.similarity) * 100).toFixed(1)}% similarity
                            </span>
                          </div>
                          
                          <div className="space-y-3">
                            {field.formd_value !== undefined && (
                              <div>
                                <span className="text-xs font-medium text-gray-500 uppercase">Form D:</span>
                                <div className="mt-1 p-2 bg-white rounded border text-sm">
                                  {field.formd_value || 'Not found'}
                                </div>
                              </div>
                            )}
                            
                            {field.invoice_value !== undefined && (
                              <div>
                                <span className="text-xs font-medium text-gray-500 uppercase">Invoice:</span>
                                <div className="mt-1 p-2 bg-white rounded border text-sm">
                                  {field.invoice_value || 'Not found'}
                                </div>
                              </div>
                            )}
                            
                            {field.bl_value !== undefined && (
                              <div>
                                <span className="text-xs font-medium text-gray-500 uppercase">BL:</span>
                                <div className="mt-1 p-2 bg-white rounded border text-sm">
                                  {field.bl_value || 'Not found'}
                                </div>
                              </div>
                            )}
                          </div>
                          
                          {field.note && (
                            <div className="mt-2 text-xs text-gray-500 italic">{field.note}</div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="bg-white rounded-xl shadow-lg p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Extracted Document Data</h3>
                  
                  <div className="space-y-4">
                    {result.data?.formd_data && (
                      <div className="border rounded-lg overflow-hidden">
                        <button
                          onClick={() => toggleSection('formd_data')}
                          className="w-full flex items-center justify-between p-4 bg-blue-50 hover:bg-blue-100 transition-colors"
                        >
                          <div className="flex items-center">
                            <FileText className="w-5 h-5 text-blue-600 mr-2" />
                            <h4 className="font-medium text-gray-900">
                              Form D Data 
                              {result.data.formd_data.pages_processed && (
                                <span className="text-sm text-gray-500 ml-2">
                                  (Pages: {result.data.formd_data.pages_processed.join(', ')})
                                </span>
                              )}
                            </h4>
                          </div>
                          {expandedSections.formd_data ? (
                            <ChevronDown className="w-5 h-5 text-gray-500" />
                          ) : (
                            <ChevronRight className="w-5 h-5 text-gray-500" />
                          )}
                        </button>
                        
                        {expandedSections.formd_data && (
                          <div className="p-4 bg-white border-t">
                            <div className="grid md:grid-cols-2 gap-4">
                              {Object.entries(result.data.formd_data).map(([key, value]) => {
                                if (key === 'products' && Array.isArray(value)) {
                                  return (
                                    <div key={key} className="col-span-2 space-y-1">
                                      <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                                        Products ({value.length}):
                                      </span>
                                      <div className="space-y-3">
                                        {value.map((product, idx) => (
                                          <div key={idx} className="p-3 bg-blue-50 rounded border border-blue-200">
                                            <div className="font-medium text-sm text-blue-900 mb-2">
                                              Product {idx + 1} (Item {product['Item Number'] || '?'})
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                              {Object.entries(product).map(([pKey, pValue]) => (
                                                <div key={pKey} className="text-xs">
                                                  <span className="font-medium text-gray-600">{pKey}:</span>
                                                  <div className="text-gray-700 bg-white p-1 rounded mt-1">
                                                    {typeof pValue === 'string' ? pValue : JSON.stringify(pValue)}
                                                  </div>
                                                </div>
                                              ))}
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  );
                                }
                                if (key === 'pages_processed' || key === 'total_products') {
                                  return null;
                                }
                                return (
                                  <div key={key} className="space-y-1">
                                    <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                                      {key}:
                                    </span>
                                    <div className="p-2 bg-gray-50 rounded border text-sm text-gray-700">
                                      {typeof value === 'string' ? value : JSON.stringify(value)}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {result.data?.invoice_data && (
                      <div className="border rounded-lg overflow-hidden">
                        <button
                          onClick={() => toggleSection('invoice_data')}
                          className="w-full flex items-center justify-between p-4 bg-green-50 hover:bg-green-100 transition-colors"
                        >
                          <div className="flex items-center">
                            <FileText className="w-5 h-5 text-green-600 mr-2" />
                            <h4 className="font-medium text-gray-900">Invoice Data</h4>
                          </div>
                          {expandedSections.invoice_data ? (
                            <ChevronDown className="w-5 h-5 text-gray-500" />
                          ) : (
                            <ChevronRight className="w-5 h-5 text-gray-500" />
                          )}
                        </button>
                        
                        {expandedSections.invoice_data && (
                          <div className="p-4 bg-white border-t">
                            <div className="grid md:grid-cols-2 gap-4">
                              {Object.entries(result.data.invoice_data).map(([key, value]) => {
                                if (key === 'products' && Array.isArray(value)) {
                                  return (
                                    <div key={key} className="col-span-2 space-y-1">
                                      <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                                        Products ({value.length}):
                                      </span>
                                      <div className="space-y-3">
                                        {value.map((product, idx) => (
                                          <div key={idx} className="p-3 bg-green-50 rounded border border-green-200">
                                            <div className="font-medium text-sm text-green-900 mb-2">
                                              Product {idx + 1}
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                              {Object.entries(product).map(([pKey, pValue]) => (
                                                <div key={pKey} className="text-xs">
                                                  <span className="font-medium text-gray-600">{pKey}:</span>
                                                  <div className="text-gray-700 bg-white p-1 rounded mt-1">
                                                    {typeof pValue === 'string' ? pValue : JSON.stringify(pValue)}
                                                  </div>
                                                </div>
                                              ))}
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  );
                                }
                                return (
                                  <div key={key} className="space-y-1">
                                    <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                                      {key}:
                                    </span>
                                    <div className="p-2 bg-gray-50 rounded border text-sm text-gray-700">
                                      {typeof value === 'string' ? value : JSON.stringify(value)}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {result.data?.bl_data && (
                      <div className="border rounded-lg overflow-hidden">
                        <button
                          onClick={() => toggleSection('bl_data')}
                          className="w-full flex items-center justify-between p-4 bg-purple-50 hover:bg-purple-100 transition-colors"
                        >
                          <div className="flex items-center">
                            <FileText className="w-5 h-5 text-purple-600 mr-2" />
                            <h4 className="font-medium text-gray-900">
                              BL Data
                              {result.data.bl_data.pages_processed && (
                                <span className="text-sm text-gray-500 ml-2">
                                  (Pages: {result.data.bl_data.pages_processed.join(', ')})
                                </span>
                              )}
                            </h4>
                          </div>
                          {expandedSections.bl_data ? (
                            <ChevronDown className="w-5 h-5 text-gray-500" />
                          ) : (
                            <ChevronRight className="w-5 h-5 text-gray-500" />
                          )}
                        </button>
                        
                        {expandedSections.bl_data && (
                          <div className="p-4 bg-white border-t">
                            <div className="grid md:grid-cols-2 gap-4">
                              {Object.entries(result.data.bl_data).map(([key, value]) => {
                                if (key === 'containers' && Array.isArray(value)) {
                                  return (
                                    <div key={key} className="col-span-2 space-y-1">
                                      <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                                        Containers ({value.length}):
                                      </span>
                                      <div className="space-y-3">
                                        {value.map((container, idx) => (
                                          <div key={idx} className="p-3 bg-purple-50 rounded border border-purple-200">
                                            <div className="font-medium text-sm text-purple-900 mb-2">
                                              Container {idx + 1}
                                            </div>
                                            <div className="grid grid-cols-2 gap-2">
                                              {Object.entries(container).map(([cKey, cValue]) => (
                                                <div key={cKey} className="text-xs">
                                                  <span className="font-medium text-gray-600">{cKey}:</span>
                                                  <div className="text-gray-700 bg-white p-1 rounded mt-1">
                                                    {typeof cValue === 'string' ? cValue : JSON.stringify(cValue)}
                                                  </div>
                                                </div>
                                              ))}
                                            </div>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  );
                                }
                                if (key === 'pages_processed' || key === 'total_containers') {
                                  return null;
                                }
                                return (
                                  <div key={key} className="space-y-1">
                                    <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">
                                      {key}:
                                    </span>
                                    <div className="p-2 bg-gray-50 rounded border text-sm text-gray-700">
                                      {typeof value === 'string' ? value : JSON.stringify(value)}
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Results Yet</h3>
                <p className="text-gray-600">Upload all three PDF files (Form D, Invoice, and BL) and click "Process Documents" to see the analysis results.</p>
                <p className="text-sm text-gray-500 mt-2">Use Advanced Settings to process multiple pages</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentProcessor;