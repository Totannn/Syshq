import streamlit as st
import pandas as pd
import json
import time
import random
from datetime import datetime, timedelta
import base64
from io import BytesIO
import re
from PIL import Image
#import pytesseract
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="SysComply - AI-Powered KYC Review",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client with API key from environment variable
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("openai", {}).get("api_key")

if not api_key:
    st.error("OpenAI API key not found. Please set it in your environment variables or Streamlit secrets.")
    st.stop()

# Initialize OpenAI client with compatibility for newer versions
try:
    client = OpenAI(api_key=api_key)
except TypeError:
    # Fallback for older OpenAI version if needed
    import openai
    openai.api_key = api_key
    client = openai


# App title and description
st.title("SysComply - AI-Powered KYC Review Assistant")
st.markdown("""
    **Streamline KYC reviews with AI-powered analysis for Nigerian financial services.**
    Upload customer documents, automatically extract data, screen against watchlists, and generate compliance reports.
""")

# Sample data for demonstration
def load_sample_data():
    # Sample PEP/Sanctions list
    sanctions_data = {
        "names": ["Adebowale Adekunle", "Chinedu Okoro", "Fatima Bello", "Ibrahim Musa", "Grace Chukwu"],
        "countries": ["Nigeria", "Nigeria", "Nigeria", "Nigeria", "Nigeria"],
        "types": ["PEP", "Sanctioned", "PEP", "Sanctioned", "PEP"],
        "reasons": ["Former government official", "Money laundering", "Political figure", "Terrorism financing", "Government contractor"]
    }
    
    # Sample customer data for timeline
    timeline_data = [
        {"date": "2023-10-15", "event": "Address updated", "source": "CRM"},
        {"date": "2023-09-22", "event": "Employment changed", "source": "Email"},
        {"date": "2023-08-05", "event": "Phone number updated", "source": "CRM"},
        {"date": "2023-07-18", "event": "Account opened", "source": "Initial KYC"}
    ]
    
    return pd.DataFrame(sanctions_data), timeline_data

# Function to simulate document processing with OCR
def process_document(uploaded_file):
    file_name = uploaded_file.name.lower()

    # Decide document type based on file extension or filename keywords
    if file_name.endswith(('.png', '.jpg', '.jpeg')):
        if "bvn" in file_name:
            doc_type = "BVN"
        elif "utility" in file_name:
            doc_type = "Utility Bill"
        else:
            doc_type = "Passport"
    elif file_name.endswith('.pdf'):
        if "cac" in file_name:
            doc_type = "CAC Certificate"
        elif "statement" in file_name:
            doc_type = "Bank Statement"
        else:
            doc_type = "PDF Document"
    else:
        doc_type = "Unknown"

    # Return simulated extracted data
    if doc_type == "Passport":
        return {
            "Document Type": "International Passport",
            "Full Name": "Adebola Johnson",
            "Date of Birth": "15-04-1985",
            "Passport Number": "A02345678",
            "Nationality": "Nigerian",
            "Gender": "Female",
            "Issue Date": "10-01-2020",
            "Expiry Date": "09-01-2030"
        }
    elif doc_type == "BVN":
        return {
            "Document Type": "BVN Slip",
            "Full Name": "Adebola Johnson",
            "BVN": "22345678901",
            "Phone Number": "08031234567",
            "Date of Birth": "15-04-1985",
            "Gender": "Female"
        }
    elif doc_type == "Utility Bill":
        return {
            "Document Type": "Utility Bill",
            "Full Name": "Adebola Johnson",
            "Address": "15 Victoria Island, Lagos, Nigeria",
            "Account Number": "UTIL-987654",
            "Service Type": "Electricity",
            "Issue Date": "05-11-2023"
        }
    elif doc_type == "CAC Certificate":
        return {
            "Document Type": "CAC Certificate",
            "Business Name": "Johnson Enterprises Ltd.",
            "RC Number": "RC-1234567",
            "Business Address": "15 Victoria Island, Lagos, Nigeria",
            "Incorporation Date": "12-06-2015",
            "Business Type": "Private Limited Company"
        }
    elif doc_type == "Bank Statement":
        return {
            "Document Type": "Bank Statement",
            "Account Name": "Adebola Johnson",
            "Account Number": "0023456789",
            "Bank Name": "First Bank of Nigeria",
            "Statement Period": "October 2023",
            "Average Balance": "₦1,245,000"
        }
    else:
        return {
            "Document Type": "Unknown",
            "Content": "Simulated extracted data"
        }


# Function to perform sanctions screening using OpenAI
def screen_against_sanctions(extracted_data, sanctions_df):
    name = extracted_data.get("Full Name", "")
    dob = extracted_data.get("Date of Birth", "")
    
    # Check for exact matches first
    exact_matches = sanctions_df[sanctions_df['names'].str.contains(name, case=False)]
    
    # If no exact matches, use OpenAI for fuzzy matching
    if exact_matches.empty:
        prompt = f"""
        Compare the name "{name}" with the following list of sanctioned individuals: {', '.join(sanctions_df['names'].tolist())}.
        Return a JSON response with:
        - match: true or false
        - matched_name: if match is true, the name that matched
        - confidence: percentage confidence of match
        - reason: if match is true, the reason for sanction
        
        Only return valid JSON.
        """
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a compliance screening assistant. Analyze names for potential matches against sanctions lists."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150
            )
            
            result = json.loads(response.choices[0].message.content)
            if result.get('match', False):
                return {
                    "match": True,
                    "matched_name": result.get('matched_name', ''),
                    "confidence": result.get('confidence', '0%'),
                    "reason": result.get('reason', ''),
                    "type": "Fuzzy Match"
                }
        except Exception as e:
            st.error(f"OpenAI API error: {str(e)}")
            # Fallback to simple matching if API call fails
            for sanctioned_name in sanctions_df['names']:
                if name.split()[0].lower() in sanctioned_name.lower() or name.split()[-1].lower() in sanctioned_name.lower():
                    match = sanctions_df[sanctions_df['names'] == sanctioned_name].iloc[0]
                    return {
                        "match": True,
                        "matched_name": sanctioned_name,
                        "confidence": "85%",
                        "reason": match['reasons'],
                        "type": match['types']
                    }
    
    elif not exact_matches.empty:
        match = exact_matches.iloc[0]
        return {
            "match": True,
            "matched_name": match['names'],
            "confidence": "100%",
            "reason": match['reasons'],
            "type": match['types']
        }
    
    return {"match": False}

# Function to generate AI summary using OpenAI
def generate_summary(extracted_data, screening_results, timeline_data):
    # Prepare prompt for OpenAI
    prompt = f"""
    As a compliance analyst, generate a concise KYC review summary based on the following data:
    
    Extracted Customer Information:
    {json.dumps(extracted_data, indent=2)}
    
    Sanctions Screening Results:
    {json.dumps(screening_results, indent=2)}
    
    Customer Timeline (Recent Changes):
    {json.dumps(timeline_data, indent=2)}
    
    Provide a professional summary that includes:
    1. Customer identification details
    2. Any red flags from sanctions screening
    3. Notable changes in customer profile
    4. Overall risk assessment (Low, Medium, High)
    5. Recommended next steps
    
    Format the response clearly with headings.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a compliance analyst preparing KYC review summaries for a financial institution."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"OpenAI API error: {str(e)}")
        # Fallback template if API call fails
        return f"""
        KYC Review Summary for {extracted_data.get('Full Name', 'Customer')}
        
        Identification Details:
        - Name: {extracted_data.get('Full Name', 'N/A')}
        - Date of Birth: {extracted_data.get('Date of Birth', 'N/A')}
        - Identification Number: {extracted_data.get('Passport Number', extracted_data.get('BVN', 'N/A'))}
        
        Sanctions Screening:
        - Result: {'Match found' if screening_results.get('match', False) else 'No matches found'}
        {('- Match Details: ' + screening_results.get('reason', '')) if screening_results.get('match', False) else ''}
        
        Risk Assessment: {'Medium' if screening_results.get('match', False) else 'Low'}
        
        Next Steps: {'Enhanced due diligence recommended' if screening_results.get('match', False) else 'Standard monitoring'}
        """

# Function to create executive dashboard
def render_executive_dashboard():
    st.header("Executive Dashboard")
    
    # Sample metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accounts Reviewed", "47", "12%")
    with col2:
        st.metric("Flagged Accounts", "8", "-3%")
    with col3:
        st.metric("Avg. Review Time", "4.2min", "-58%")
    with col4:
        st.metric("Compliance Score", "92%", "5%")
    
    # Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Review Status")
        status_data = pd.DataFrame({
            'Status': ['Completed', 'In Progress', 'Flagged', 'Pending'],
            'Count': [35, 12, 8, 22]
        })
        st.bar_chart(status_data, x='Status', y='Count')
    
    with col2:
        st.subheader("Review Time Trends")
        time_data = pd.DataFrame({
            'Week': [f'Wk {i}' for i in range(1, 9)],
            'Time (min)': [10.2, 9.5, 8.7, 7.8, 6.5, 5.3, 4.8, 4.2]
        })
        st.line_chart(time_data, x='Week', y='Time (min)')
    
    # Recent activities
    st.subheader("Recent Flagged Cases")
    flagged_cases = pd.DataFrame({
        'Customer': ['Adebola Johnson', 'Chinedu Okoro', 'Fatima Bello', 'Emeka Nwosu'],
        'Date': ['2023-11-05', '2023-11-04', '2023-11-03', '2023-11-02'],
        'Risk Level': ['Medium', 'High', 'Medium', 'Low'],
        'Reason': ['PEP Match', "Sanctions Match", "Address Discrepancy", "Document Expiry"]
    })
    st.dataframe(flagged_cases, use_container_width=True)

# Main app logic
def main():
    # Load sample data
    sanctions_df, sample_timeline = load_sample_data()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Compliance Analyst", "Executive Dashboard"])
    
    with tab1:
        st.header("KYC Document Review")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload KYC Documents", 
            type=['pdf', 'png', 'jpg', 'jpeg'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            # Process each uploaded file
            extracted_data_list = []
            screening_results_list = []
            
            for uploaded_file in uploaded_files:
                with st.expander(f"Document: {uploaded_file.name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Uploaded Document")
                        if uploaded_file.type.startswith('image'):
                            st.image(uploaded_file)
                        else:
                            st.info(f"PDF document: {uploaded_file.name}")
                    
                    # Process document
                    with st.spinner("Extracting data..."):
                        extracted_data = process_document(uploaded_file)
                        extracted_data_list.append(extracted_data)
                        
                        with col2:
                            st.subheader("Extracted Data")
                            for key, value in extracted_data.items():
                                st.write(f"**{key}:** {value}")
            
            # If we have extracted data, show verification and screening
            if extracted_data_list:
                # Combine extracted data (in a real app, this would be more sophisticated)
                combined_data = {}
                for data in extracted_data_list:
                    combined_data.update(data)
                
                st.divider()
                
                # Identity verification panel
                st.subheader("Identity Verification")
                col1, col2, col3 = st.columns(3)
                
                # Simulate CRM data for verification
                crm_data = {
                    "Full Name": "Adebola Johnson",
                    "Date of Birth": "15-04-1985",
                    "Address": "15 Victoria Island, Lagos, Nigeria"
                }
                
                verification_issues = []
                
                with col1:
                    st.write("**Extracted Data**")
                    for field in ["Full Name", "Date of Birth", "Address"]:
                        value = combined_data.get(field, "N/A")
                        st.write(f"{field}: {value}")
                
                with col2:
                    st.write("**CRM Data**")
                    for field, value in crm_data.items():
                        st.write(f"{field}: {value}")
                
                with col3:
                    st.write("**Verification Status**")
                    for field, expected_value in crm_data.items():
                        actual_value = combined_data.get(field, "")
                        if actual_value and actual_value != expected_value:
                            st.error(f"❌ {field} mismatch")
                            verification_issues.append(f"{field} mismatch")
                        elif actual_value:
                            st.success(f"✅ {field} matches")
                        else:
                            st.warning(f"⚠️ {field} not found")
                
                # Sanctions screening
                st.subheader("Sanctions Screening")
                with st.spinner("Screening against watchlists..."):
                    screening_results = screen_against_sanctions(combined_data, sanctions_df)
                    
                    if screening_results.get("match", False):
                        st.error(f"**Risk Alert**: Potential match with {screening_results.get('type', 'sanctioned')} individual")
                        st.write(f"**Matched Name**: {screening_results.get('matched_name', 'N/A')}")
                        st.write(f"**Confidence**: {screening_results.get('confidence', 'N/A')}")
                        st.write(f"**Reason**: {screening_results.get('reason', 'N/A')}")
                    else:
                        st.success("No matches found in sanctions/PEP databases")
                
                # Change tracker timeline
                st.subheader("Customer Profile Timeline")
                for event in sample_timeline:
                    st.write(f"**{event['date']}** ({event['source']}): {event['event']}")
                
                # AI Summary Generator
                st.subheader("Compliance Summary")
                with st.spinner("Generating AI summary..."):
                    summary = generate_summary(combined_data, screening_results, sample_timeline)
                    st.text_area("Review Summary", summary, height=300)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✅ Approve KYC", type="primary"):
                        st.success("KYC approved successfully!")
                with col2:
                    if st.button("⛔ Flag for Review"):
                        st.error("KYC flagged for manual review")
                with col3:
                    if st.button("📋 Copy to Report"):
                        st.info("Summary copied to clipboard")
            
            else:
                st.warning("No data could be extracted from the uploaded documents.")
        
        else:
            # Show sample documents and instructions
            st.info("""
            **Demo Instructions:**
            1. Upload sample KYC documents (PDF or images)
            2. SysComply will automatically extract data
            3. Review identity verification results
            4. Check sanctions screening outcomes
            5. Generate a compliance summary
            
            **Supported documents:** Passport, BVN, Utility Bills, CAC Certificates, Bank Statements
            """)
            
            # Show sample documents
            st.subheader("Sample Documents for Testing")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**International Passport**")
                st.image("https://via.placeholder.com/150x200.png?text=Sample+Passport", width=150)
            with col2:
                st.write("**BVN Slip**")
                st.image("https://via.placeholder.com/150x200.png?text=Sample+BVN", width=150)
            with col3:
                st.write("**Utility Bill**")
                st.image("https://via.placeholder.com/150x200.png?text=Utility+Bill", width=150)
    
    with tab2:
        render_executive_dashboard()

if __name__ == "__main__":
    main()
