import React, { useState } from 'react';
import { Upload, FileText, CheckCircle, XCircle, AlertCircle, Loader2, Download, ChevronDown, ChevronRight } from 'lucide-react';

const DocumentProcessor = () => {
  const [formDFile, setFormDFile] = useState(null);
  const [invoiceFile, setInvoiceFile] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expandedSections, setExpandedSections] = useState({});

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
      } else {
        setInvoiceFile(file);
      }
      setError(null);
    } else {
      setError('Please select a valid PDF file');
    }
  };

  const handleSubmit = async () => {
    if (!formDFile || !invoiceFile) {
      setError('Please select both Form D and Invoice PDF files');
      return;
    }

    setIsProcessing(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('formd_pdf', formDFile);
    formData.append('invoice_pdf', invoiceFile);

    try {
      const response = await fetch('http://localhost:8000/process-documents-simple', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
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

  const ValueComparisonCard = ({ title, formDValue, invoiceValue, similarity, fieldName }) => (
    <div className="border rounded-lg p-4 bg-gray-50">
      <div className="flex justify-between items-start mb-3">
        <h5 className="font-medium text-gray-900">{title}</h5>
        <span className="text-sm text-blue-700 bg-blue-100 px-2 py-1 rounded">
          {(similarity * 100).toFixed(1)}% match
        </span>
      </div>
      
      <div className="space-y-3">
        <div>
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Form D Value:</span>
          <div className="mt-1 p-2 bg-white rounded border text-sm">
            {formDValue || 'Not found'}
          </div>
        </div>
        
        <div>
          <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Invoice Value:</span>
          <div className="mt-1 p-2 bg-white rounded border text-sm">
            {invoiceValue || 'Not found'}
          </div>
        </div>
      </div>
    </div>
  );

  const ValidationMatchCard = ({ match, type }) => (
    <div className="border rounded-lg p-4 bg-gray-50">
      {type === 'product' ? (
        <div className="space-y-3">
          <div className="flex justify-between items-start">
            <h5 className="font-medium text-gray-900">Product Match</h5>
            <div className="flex space-x-2">
              <span className="text-xs text-blue-700 bg-blue-100 px-2 py-1 rounded">
                HS: {(match.hs_code_similarity * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-green-700 bg-green-100 px-2 py-1 rounded">
                Desc: {(match.description_similarity * 100).toFixed(1)}%
              </span>
            </div>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Form D HS Code:</span>
              <div className="mt-1 p-2 bg-white rounded border text-sm font-mono">
                {match.formd_hs_code || 'Not found'}
              </div>
            </div>
            
            <div>
              <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">CSV HS Code:</span>
              <div className="mt-1 p-2 bg-white rounded border text-sm font-mono">
                {match.csv_hs_code || 'Not found'}
              </div>
            </div>
          </div>
          
          <div>
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Form D Description:</span>
            <div className="mt-1 p-2 bg-white rounded border text-sm">
              {match.formd_description || 'Not found'}
            </div>
          </div>
          
          <div>
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">CSV Form D Description:</span>
            <div className="mt-1 p-2 bg-white rounded border text-sm">
              {match.csv_form_d_description || 'Not found'}
            </div>
          </div>
          
          
          
          {match.match_reason && (
            <div className="text-xs text-gray-600 italic">
              Match reason: {match.match_reason}
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          <div className="flex justify-between items-start">
            <h5 className="font-medium text-gray-900">Company Match</h5>
            <div className="flex space-x-2">
              <span className="text-xs text-blue-700 bg-blue-100 px-2 py-1 rounded">
                Name: {(match.company_name_similarity * 100).toFixed(1)}%
              </span>
              <span className="text-xs text-green-700 bg-green-100 px-2 py-1 rounded">
                Address: {match.address_match ? 'Match' : 'No Match'}
              </span>
            </div>
          </div>
          
          <div>
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Form D Consignee Name:</span>
            <div className="mt-1 p-2 bg-white rounded border text-sm">
              {match.formd_consignee_name || 'Not found'}
            </div>
          </div>
          
          <div>
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">CSV Company Name:</span>
            <div className="mt-1 p-2 bg-white rounded border text-sm">
              {match.csv_company_name || 'Not found'}
            </div>
          </div>
          
          <div>
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Form D Address:</span>
            <div className="mt-1 p-2 bg-white rounded border text-sm">
              {match.formd_consignee_address || 'Not found'}
            </div>
          </div>
          
          <div>
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Combined Form D Info:</span>
            <div className="mt-1 p-2 bg-white rounded border text-sm">
              {match.combined_formd_info || 'Not found'}
            </div>
          </div>
          
          <div>
            <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">CSV Address:</span>
            <div className="mt-1 p-2 bg-white rounded border text-sm">
              {match.csv_address || 'Not found'}
            </div>
          </div>
          
          {match.address_similarity !== undefined && (
            <div className="text-xs text-gray-600">
              Address similarity: {(match.address_similarity * 100).toFixed(1)}%
            </div>
          )}
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Document Processor</h1>
          <p className="text-lg text-gray-600">Upload Form D and Invoice PDFs for AI-powered analysis</p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-3 gap-6">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl shadow-lg p-6">
              <h2 className="text-xl font-semibold text-gray-900 mb-4 flex items-center">
                <Upload className="w-5 h-5 mr-2" />
                Upload Documents
              </h2>

              <div className="space-y-4">
                {/* Form D Upload */}
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
                      className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer hover:bg-gray-50 hover:border-gray-400 transition-colors"
                    >
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <FileText className="w-8 h-8 text-gray-400 mb-2" />
                        <p className="text-sm text-gray-500">
                          {formDFile ? formDFile.name : 'Click to upload Form D'}
                        </p>
                      </div>
                    </label>
                  </div>
                </div>

                {/* Invoice Upload */}
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
                      className="flex flex-col items-center justify-center w-full h-32 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer hover:bg-gray-50 hover:border-gray-400 transition-colors"
                    >
                      <div className="flex flex-col items-center justify-center pt-5 pb-6">
                        <FileText className="w-8 h-8 text-gray-400 mb-2" />
                        <p className="text-sm text-gray-500">
                          {invoiceFile ? invoiceFile.name : 'Click to upload Invoice'}
                        </p>
                      </div>
                    </label>
                  </div>
                </div>

                {/* Submit Button */}
                <button
                  onClick={handleSubmit}
                  disabled={!formDFile || !invoiceFile || isProcessing}
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

              {/* Error Display */}
              {error && (
                <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-lg">
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* Results Section */}
          <div className="lg:col-span-2">
            {result ? (
              <div className="space-y-6">
                {/* Summary Card */}
                <div className="bg-white rounded-xl shadow-lg p-6">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-semibold text-gray-900">Analysis Summary</h2>
                    <button
                      onClick={downloadResults}
                      className="flex items-center px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                    >
                      <Download className="w-4 h-4 mr-2" />
                      Download
                    </button>
                  </div>

                  {result.data?.overall_assessment && (
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
                      </div>
                    </div>
                  )}

                  {result.data?.overall_assessment?.summary && (
                    <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                      <p className="text-sm text-gray-700">{result.data.overall_assessment.summary}</p>
                    </div>
                  )}
                </div>

                {/* Validation Details */}
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
                      {/* Product Validation */}
                      {result.data.validation.product_validation && (
                        <div className="border rounded-lg p-4">
                          <div className="flex items-center justify-between mb-2">
                            <div className="flex items-center">
                              {getStatusIcon(result.data.validation.product_validation.found_matches ? 'PASS' : 'FAIL')}
                              <h4 className="ml-2 font-medium">Product Validation</h4>
                            </div>
                            <span className="text-xs text-gray-500">
                              {result.data.validation.product_validation.matches?.length || 0} matches found
                            </span>
                          </div>
                          
                          {expandedSections.validation && result.data.validation.product_validation.matches?.length > 0 && (
                            <div className="mt-4 space-y-4">
                              {result.data.validation.product_validation.matches.slice(0, 3).map((match, idx) => (
                                <ValidationMatchCard key={idx} match={match} type="product" />
                              ))}
                            </div>
                          )}
                        </div>
                      )}

                      {/* Company Validation */}
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
                            <div className="mt-4 space-y-4">
                              {result.data.validation.company_validation.matches.slice(0, 3).map((match, idx) => (
                                <ValidationMatchCard key={idx} match={match} type="company" />
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Matching Fields with Values */}
                {result.data?.comparison?.matching_fields?.length > 0 && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-semibold text-gray-900">Matching Fields & Values</h3>
                      <span className="text-sm text-gray-500">
                        {result.data.comparison.matching_fields.length} fields matched
                      </span>
                    </div>
                    
                    <div className="space-y-4">
                      {result.data.comparison.matching_fields.map((field, idx) => (
                        <ValueComparisonCard
                          key={idx}
                          title={field.field}
                          formDValue={field.formd_value}
                          invoiceValue={field.invoice_value}
                          similarity={field.similarity}
                          fieldName={field.field}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* Discrepancies */}
                {result.data?.comparison?.discrepancies?.length > 0 && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Discrepancies</h3>
                    <div className="space-y-4">
                      {result.data.comparison.discrepancies.map((field, idx) => (
                        <div key={idx} className="border border-red-200 rounded-lg p-4 bg-red-50">
                          <div className="flex justify-between items-start mb-3">
                            <h5 className="font-medium text-gray-900">{field.field}</h5>
                            <span className="text-sm text-red-700 bg-red-200 px-2 py-1 rounded">
                              {(field.similarity * 100).toFixed(1)}% similarity
                            </span>
                          </div>
                          
                          <div className="space-y-3">
                            <div>
                              <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Form D Value:</span>
                              <div className="mt-1 p-2 bg-white rounded border text-sm">
                                {field.formd_value || 'Not found'}
                              </div>
                            </div>
                            
                            <div>
                              <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">Invoice Value:</span>
                              <div className="mt-1 p-2 bg-white rounded border text-sm">
                                {field.invoice_value || 'Not found'}
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Recommendations */}
                {result.data?.overall_assessment?.recommendations?.length > 0 && (
                  <div className="bg-white rounded-xl shadow-lg p-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Recommendations</h3>
                    <ul className="space-y-2">
                      {result.data.overall_assessment.recommendations.map((rec, idx) => (
                        <li key={idx} className="flex items-start">
                          <AlertCircle className="w-4 h-4 mt-1 mr-2 text-amber-500 flex-shrink-0" />
                          <span className="text-sm text-gray-700">{rec}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <div className="bg-white rounded-xl shadow-lg p-8 text-center">
                <FileText className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Results Yet</h3>
                <p className="text-gray-600">Upload both PDF files and click "Process Documents" to see the analysis results.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentProcessor;