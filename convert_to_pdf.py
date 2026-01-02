#!/usr/bin/env python3
"""
Convert PROJECT_REPORT.md to PDF format using reportlab
"""
import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def parse_markdown(md_content):
    """Parse markdown content into structured elements"""
    lines = md_content.split('\n')
    elements = []
    in_code_block = False
    code_block_content = []
    code_language = ""
    
    for line in lines:
        # Code blocks
        if line.strip().startswith('```'):
            if in_code_block:
                # End of code block
                if code_block_content:
                    elements.append(('code', '\n'.join(code_block_content), code_language))
                code_block_content = []
                code_language = ""
                in_code_block = False
            else:
                # Start of code block
                in_code_block = True
                code_language = line.strip()[3:].strip()
            continue
        
        if in_code_block:
            code_block_content.append(line)
            continue
        
        # Headers
        if line.startswith('# '):
            elements.append(('h1', line[2:].strip()))
        elif line.startswith('## '):
            elements.append(('h2', line[3:].strip()))
        elif line.startswith('### '):
            elements.append(('h3', line[4:].strip()))
        elif line.startswith('#### '):
            elements.append(('h4', line[5:].strip()))
        # Horizontal rule
        elif line.strip() == '---':
            elements.append(('hr',))
        # List items (numbered or bulleted)
        elif re.match(r'^\d+\.\s+', line) or line.strip().startswith('- ') or line.strip().startswith('* '):
            # Extract list item content (remove marker)
            content = re.sub(r'^\d+\.\s+', '', line)
            content = re.sub(r'^[-*]\s+', '', content)
            elements.append(('list_item', content.strip()))
        # Empty line
        elif not line.strip():
            elements.append(('spacer',))
        # Regular paragraph
        else:
            elements.append(('p', line))
    
    return elements

def markdown_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF"""
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Parse markdown
    elements = parse_markdown(md_content)
    
    # Create PDF document
    doc = SimpleDocTemplate(
        str(pdf_file),
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#1a1a1a'),
        spaceAfter=20,
        spaceBefore=30,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=HexColor('#1a1a1a'),
        spaceAfter=15,
        spaceBefore=25,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold',
        borderWidth=0,
        borderPadding=0
    )
    
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=20,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=14,
        textColor=HexColor('#34495e'),
        spaceAfter=10,
        spaceBefore=15,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    h4_style = ParagraphStyle(
        'CustomH4',
        parent=styles['Heading4'],
        fontSize=12,
        textColor=HexColor('#555555'),
        spaceAfter=8,
        spaceBefore=12,
        alignment=TA_LEFT,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor('#333333'),
        spaceAfter=10,
        alignment=TA_JUSTIFY,
        leading=16
    )
    
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=9,
        textColor=HexColor('#333333'),
        backColor=HexColor('#f4f4f4'),
        leftIndent=20,
        rightIndent=20,
        spaceAfter=12,
        spaceBefore=12,
        fontName='Courier',
        alignment=TA_LEFT
    )
    
    # Build story (content)
    story = []
    
    def clean_text(text):
        """Clean and escape text for reportlab"""
        # First escape existing & to avoid conflicts
        text = text.replace('&', '&amp;')
        
        # Handle HTML bold/strong tags if they exist (convert to reportlab format)
        text = re.sub(r'<b>(.*?)</b>', r'__BOLD__\1__/BOLD__', text, flags=re.IGNORECASE)
        text = re.sub(r'<strong>(.*?)</strong>', r'__BOLD__\1__/BOLD__', text, flags=re.IGNORECASE)
        
        # Handle markdown bold formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'__BOLD__\1__/BOLD__', text)
        
        # Handle markdown italic (but not if it's part of bold)
        # Only match single asterisks that aren't part of double asterisks
        text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'__ITALIC__\1__/ITALIC__', text)
        
        # Handle inline code
        text = re.sub(r'`([^`]+?)`', r'__CODE__\1__/CODE__', text)
        
        # Now escape remaining HTML entities (but preserve our placeholders)
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Restore formatting tags as ReportLab HTML
        text = text.replace('__BOLD__', '<b>')
        text = text.replace('__/BOLD__', '</b>')
        text = text.replace('__ITALIC__', '<i>')
        text = text.replace('__/ITALIC__', '</i>')
        text = text.replace('__CODE__', '<font name="Courier" color="#c7254e">')
        text = text.replace('__/CODE__', '</font>')
        
        return text
    
    for elem_type, *args in elements:
        if elem_type == 'h1':
            text = clean_text(args[0])
            story.append(Paragraph(text, h1_style))
            story.append(Spacer(1, 0.2*inch))
        elif elem_type == 'h2':
            text = clean_text(args[0])
            story.append(Paragraph(text, h2_style))
            story.append(Spacer(1, 0.15*inch))
        elif elem_type == 'h3':
            text = clean_text(args[0])
            story.append(Paragraph(text, h3_style))
            story.append(Spacer(1, 0.1*inch))
        elif elem_type == 'h4':
            text = clean_text(args[0])
            story.append(Paragraph(text, h4_style))
            story.append(Spacer(1, 0.08*inch))
        elif elem_type == 'list_item':
            text = args[0].strip()
            if text:
                text = clean_text(text)
                # Add bullet point
                story.append(Paragraph(f"• {text}", normal_style))
        elif elem_type == 'p':
            text = args[0].strip()
            if text:
                text = clean_text(text)
                story.append(Paragraph(text, normal_style))
        elif elem_type == 'code':
            code_text = args[0]
            # Escape HTML in code
            code_text = code_text.replace('&', '&amp;')
            code_text = code_text.replace('<', '&lt;')
            code_text = code_text.replace('>', '&gt;')
            story.append(Preformatted(code_text, code_style))
        elif elem_type == 'hr':
            story.append(Spacer(1, 0.3*inch))
        elif elem_type == 'spacer':
            story.append(Spacer(1, 0.1*inch))
    
    # Build PDF
    print(f"Converting {md_file} to PDF...")
    doc.build(story)
    print(f"PDF created successfully: {pdf_file}")

if __name__ == "__main__":
    md_file = Path("PROJECT_REPORT.md")
    pdf_file = Path("PROJECT_REPORT.pdf")
    
    if not md_file.exists():
        print(f"Error: {md_file} not found!")
        exit(1)
    
    try:
        markdown_to_pdf(md_file, pdf_file)
        print(f"\nSuccess! PDF report saved as: {str(pdf_file.absolute())}")
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
