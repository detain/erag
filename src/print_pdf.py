# -*- coding: utf-8 -*-
# Standard library imports
import os
from datetime import datetime
import re

# Third-party imports
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Frame, PageTemplate, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus.paragraph import ParaLines

# Define RGB values for custom colors
DARK_BLUE_RGB = (34/255, 34/255, 59/255)
MEDIUM_BLUE_RGB = (43/255, 116/255, 238/255)

class PDFReportGenerator:
    def __init__(self, output_folder, llm_name, project_name):
        self.output_folder = output_folder
        self.llm_name = llm_name
        self.project_name = project_name
        self.report_title = None
        self.styles = self._create_styles()

    def create_enhanced_pdf_report(self, findings, pdf_content, image_data, filename="report", report_title=None):
        self.report_title = report_title or f"Analysis Report for {self.project_name}"
        pdf_file = os.path.join(self.output_folder, f"{filename}.pdf")
        doc = SimpleDocTemplate(pdf_file, pagesize=A4)

        elements = []

        # Cover page
        elements.extend(self._create_cover_page(doc))

        # Table of Contents
        elements.append(Paragraph("Table of Contents", self.styles['Heading1']))
        
        # Add key findings entry
        elements.append(Paragraph("Key Findings", self.styles['TOCEntry']))
        elements.append(Spacer(1, 6))
        
        # Add content entries
        for i, (analysis_type, _, _) in enumerate(pdf_content):
            elements.append(Paragraph(f"{i+1}. {analysis_type}", self.styles['TOCEntry']))
            elements.append(Spacer(1, 4))  # Small space between entries
        
        elements.append(PageBreak())

        # Key Findings
        if findings:
            elements.append(Paragraph("Key Findings", self.styles['Heading1']))
            for finding in findings:
                elements.extend(self._text_to_reportlab(finding))
            elements.append(PageBreak())

        # Main content
        for i, (analysis_type, image_paths, interpretation) in enumerate(pdf_content):
            # Add section numbering
            elements.append(Paragraph(f"{i+1}. {analysis_type}", self.styles['Heading1']))
            elements.extend(self._text_to_reportlab(interpretation))

            # Add images for this analysis type
            for description, img_path in image_paths:
                if os.path.exists(img_path):
                    img = Image(img_path)
                    available_width = doc.width
                    aspect = img.drawHeight / img.drawWidth
                    img.drawWidth = available_width
                    img.drawHeight = available_width * aspect
                    elements.append(img)
                    elements.append(Paragraph(description, self.styles['Caption']))
                    elements.append(Spacer(1, 12))

            elements.append(PageBreak())

        try:
            doc.build(elements, onFirstPage=self._add_header_footer, onLaterPages=self._add_header_footer)
            print(f"PDF report saved to {pdf_file}")
            return pdf_file
        except Exception as e:
            print(f"Error building PDF: {str(e)}")
            return None

    def _create_styles(self):
        styles = getSampleStyleSheet()
        
        # Modify existing styles
        styles['Title'].fontSize = 24
        styles['Title'].alignment = TA_CENTER
        styles['Title'].textColor = colors.white
        styles['Title'].backColor = colors.Color(*MEDIUM_BLUE_RGB)
        styles['Title'].spaceAfter = 12
        styles['Title'].spaceBefore = 12
        styles['Title'].leading = 30

        styles['Heading1'].fontSize = 18
        styles['Heading1'].alignment = TA_JUSTIFY
        styles['Heading1'].spaceAfter = 12
        styles['Heading1'].spaceBefore = 8
        styles['Heading1'].textColor = colors.Color(*MEDIUM_BLUE_RGB)

        # Modify or add Heading2 style
        if 'Heading2' in styles:
            styles['Heading2'].fontSize = 16
            styles['Heading2'].textColor = colors.Color(*DARK_BLUE_RGB)
            styles['Heading2'].spaceBefore = 8
            styles['Heading2'].spaceAfter = 6
        else:
            styles.add(ParagraphStyle(
                name='Heading2',
                parent=styles['Heading1'],
                fontSize=16,
                textColor=colors.Color(*DARK_BLUE_RGB),
                spaceBefore=8,
                spaceAfter=6
            ))
        
        # Modify or add Heading3 style
        if 'Heading3' in styles:
            styles['Heading3'].fontSize = 14
            styles['Heading3'].textColor = colors.Color(*DARK_BLUE_RGB, 0.8)
            styles['Heading3'].spaceBefore = 6
            styles['Heading3'].spaceAfter = 4
        else:
            styles.add(ParagraphStyle(
                name='Heading3',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.Color(*DARK_BLUE_RGB, 0.8),
                spaceBefore=6,
                spaceAfter=4
            ))

        # Modify or add Heading4 style
        if 'Heading4' in styles:
            styles['Heading4'].fontSize = 12
            styles['Heading4'].fontName = 'Helvetica-Bold'
            styles['Heading4'].textColor = colors.black
            styles['Heading4'].spaceBefore = 4
            styles['Heading4'].spaceAfter = 2
            styles['Heading4'].leading = 14
        else:
            styles.add(ParagraphStyle(
                name='Heading4',
                parent=styles['Heading3'],
                fontSize=12,
                fontName='Helvetica-Bold',
                textColor=colors.black,
                spaceBefore=4,
                spaceAfter=2,
                leading=14
            ))

        # Modify Normal style
        styles['Normal'].fontSize = 10
        styles['Normal'].alignment = TA_JUSTIFY
        styles['Normal'].spaceAfter = 6
        styles['Normal'].textColor = colors.black
        styles['Normal'].leading = 14  # Better line spacing

        # Add or modify BulletPoint style
        if 'BulletPoint' in styles:
            styles['BulletPoint'].bulletIndent = 20
            styles['BulletPoint'].leftIndent = 40
            styles['BulletPoint'].firstLineIndent = -20
            styles['BulletPoint'].spaceBefore = 2
            styles['BulletPoint'].spaceAfter = 2
        else:
            styles.add(ParagraphStyle(
                name='BulletPoint',
                parent=styles['Normal'],
                bulletIndent=20,
                leftIndent=40,
                firstLineIndent=-20,
                spaceBefore=2,
                spaceAfter=2
            ))
        
        # Add or modify SubBulletPoint style
        if 'SubBulletPoint' in styles:
            styles['SubBulletPoint'].bulletIndent = 40
            styles['SubBulletPoint'].leftIndent = 60
            styles['SubBulletPoint'].firstLineIndent = -20
            styles['SubBulletPoint'].spaceBefore = 1
            styles['SubBulletPoint'].spaceAfter = 1
        else:
            styles.add(ParagraphStyle(
                name='SubBulletPoint',
                parent=styles['BulletPoint'],
                bulletIndent=40,
                leftIndent=60,
                firstLineIndent=-20,
                spaceBefore=1,
                spaceAfter=1
            ))

        # Add or modify TOCEntry style
        if 'TOCEntry' in styles:
            styles['TOCEntry'].fontSize = 11
            styles['TOCEntry'].leftIndent = 20
            styles['TOCEntry'].firstLineIndent = -20
            styles['TOCEntry'].spaceBefore = 2
            styles['TOCEntry'].spaceAfter = 2
        else:
            styles.add(ParagraphStyle(
                name='TOCEntry',
                parent=styles['Normal'],
                fontSize=11,
                leftIndent=20,
                firstLineIndent=-20,
                spaceBefore=2,
                spaceAfter=2
            ))

        # Add or modify Caption style
        if 'Caption' in styles:
            styles['Caption'].fontSize = 8
            styles['Caption'].alignment = TA_CENTER
            styles['Caption'].spaceAfter = 6
            styles['Caption'].textColor = colors.Color(*DARK_BLUE_RGB)
            styles['Caption'].fontName = 'Helvetica-Bold'
        else:
            styles.add(ParagraphStyle(
                name='Caption',
                parent=styles['Normal'],
                fontSize=8,
                alignment=TA_CENTER,
                spaceAfter=6,
                textColor=colors.Color(*DARK_BLUE_RGB),
                fontName='Helvetica-Bold'
            ))
            
        # Add or modify SectionTitle style
        if 'SectionTitle' in styles:
            styles['SectionTitle'].fontSize = 11
            styles['SectionTitle'].fontName = 'Helvetica-Bold'
            styles['SectionTitle'].spaceBefore = 6
            styles['SectionTitle'].spaceAfter = 2
        else:
            styles.add(ParagraphStyle(
                name='SectionTitle',
                parent=styles['Normal'],
                fontSize=11,
                fontName='Helvetica-Bold',
                spaceBefore=6,
                spaceAfter=2
            ))
            
        return styles

    def _create_cover_page(self, doc):
        def draw_background(canvas, doc):
            canvas.saveState()
            canvas.setFillColor(colors.Color(*MEDIUM_BLUE_RGB))
            canvas.rect(0, 0, doc.pagesize[0], doc.pagesize[1], fill=1)
            canvas.restoreState()

        cover_frame = Frame(
            doc.leftMargin, 
            doc.bottomMargin, 
            doc.width, 
            doc.height,
            id='CoverFrame'
        )
        cover_template = PageTemplate(id='CoverPage', frames=[cover_frame], onPage=draw_background)
        doc.addPageTemplates([cover_template])

        elements = []
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph(self.report_title, self.styles['Title']))
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d')}", self.styles['Normal']))
        elements.append(Paragraph(f"AI-powered analysis by ERAG using {self.llm_name}", self.styles['Normal']))
        elements.append(PageBreak())

        # Add a normal template for subsequent pages
        normal_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='NormalFrame')
        normal_template = PageTemplate(id='NormalPage', frames=[normal_frame], onPage=self._add_header_footer)
        doc.addPageTemplates([normal_template])

        return elements

    def _add_header_footer(self, canvas, doc):
        canvas.saveState()
        
        # Header
        canvas.setFillColor(colors.Color(*DARK_BLUE_RGB))
        canvas.setFont('Helvetica-Bold', 8)
        canvas.drawString(inch, doc.pagesize[1] - 0.5*inch, self.report_title)
        
        # Footer
        canvas.setFillColor(colors.Color(*DARK_BLUE_RGB))
        canvas.setFont('Helvetica', 8)
        canvas.drawString(inch, 0.5 * inch, f"Page {doc.page}")
        canvas.drawRightString(doc.pagesize[0] - inch, 0.5 * inch, "Powered by ERAG")

        canvas.restoreState()

    def _text_to_reportlab(self, text):
        """Convert text with Markdown-like formatting to ReportLab elements with improved formatting."""
        elements = []
        paragraphs = []
        current_paragraph = []
        
        # Split the text into individual lines
        lines = text.split('\n')
        
        in_list = False
        list_items = []
        list_level = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip empty lines at the beginning
            if not line.strip() and not current_paragraph:
                i += 1
                continue
            
            # Handle Markdown headings
            if line.startswith('# '):
                # If there's a current paragraph, add it to the paragraphs list
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Add the heading
                paragraphs.append(('heading1', line[2:]))
                
                # Add extra spacing after headings
                paragraphs.append('')
                
            elif line.startswith('## '):
                # If there's a current paragraph, add it to the paragraphs list
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Add the heading
                paragraphs.append(('heading2', line[3:]))
                
                # Add extra spacing after headings
                paragraphs.append('')
                
            elif line.startswith('### '):
                # If there's a current paragraph, add it to the paragraphs list
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Add the heading
                paragraphs.append(('heading3', line[4:]))
                
                # Add extra spacing after headings
                paragraphs.append('')
                
            elif line.startswith('#### '):
                # If there's a current paragraph, add it to the paragraphs list
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Add the heading
                paragraphs.append(('heading4', line[5:]))
                
                # Add extra spacing after headings
                paragraphs.append('')
            
            # Handle list items
            elif line.strip().startswith('* ') or line.strip().startswith('- '):
                # If there's a current paragraph, add it
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Determine list level based on indentation
                indent = len(line) - len(line.lstrip())
                if indent >= 4:
                    list_level = 1  # Sub-bullet
                else:
                    list_level = 0  # Main bullet
                
                # Start/continue a list
                if not in_list:
                    # If we're starting a new list, add extra space before it
                    if paragraphs and paragraphs[-1] != '':
                        paragraphs.append('')
                    in_list = True
                    list_items = []
                
                # Add this item to the list with its level
                list_items.append((list_level, line.strip()[2:]))
                
                # Check if the next line is also a list item
                if i + 1 < len(lines) and (lines[i+1].strip().startswith('* ') or lines[i+1].strip().startswith('- ')):
                    # If the next line is also a list item, continue
                    i += 1
                    continue
                else:
                    # End of list, add it as a special item
                    paragraphs.append(('bullet_list', list_items))
                    in_list = False
            
            # Handle empty lines as paragraph breaks
            elif not line.strip():
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Only add empty paragraph if the last one wasn't already empty
                if paragraphs and paragraphs[-1] != '':
                    paragraphs.append('')
            
            # Handle bold section titles (e.g., "**Title:**")
            elif line.strip().startswith('**') and '**:' in line:
                # If there's a current paragraph, add it
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                
                # Add as a section title
                paragraphs.append(('section_title', line.strip()))
                
                # Add space after the title
                paragraphs.append('')
            
            # Regular text - add to the current paragraph
            else:
                # Replace markdown formatting with HTML equivalents for better display
                # Bold
                line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                # Italic
                line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line)
                
                current_paragraph.append(line)
            
            i += 1
        
        # Add any remaining paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Convert paragraphs to ReportLab elements
        for p in paragraphs:
            if not p:
                # Empty paragraph - add a spacer
                elements.append(Spacer(1, 6))
            elif isinstance(p, tuple):
                if p[0] == 'heading1':
                    elements.append(Paragraph(p[1], self.styles['Heading1']))
                    elements.append(Spacer(1, 12))
                elif p[0] == 'heading2':
                    elements.append(Paragraph(p[1], self.styles['Heading2']))
                    elements.append(Spacer(1, 8))
                elif p[0] == 'heading3':
                    elements.append(Paragraph(p[1], self.styles['Heading3']))
                    elements.append(Spacer(1, 6))
                elif p[0] == 'heading4':
                    elements.append(Paragraph(p[1], self.styles['Heading4']))
                    elements.append(Spacer(1, 4))
                elif p[0] == 'section_title':
                    # Clean up the section title formatting
                    title = p[1].replace('**', '')
                    elements.append(Paragraph(title, self.styles['SectionTitle']))
                elif p[0] == 'bullet_list':
                    # Add each bullet item as a paragraph with proper indentation
                    for level, item in p[1]:
                        if level == 0:
                            # Main bullet
                            elements.append(Paragraph(f"• {item}", self.styles['BulletPoint']))
                        else:
                            # Sub-bullet
                            elements.append(Paragraph(f"   ○ {item}", self.styles['SubBulletPoint']))
                    elements.append(Spacer(1, 6))  # Add space after list
            else:
                try:
                    # Try to create a Paragraph object with proper formatting
                    para = Paragraph(p, self.styles['Normal'])
                    elements.append(para)
                except Exception as e:
                    # If there's an error, clean the text and try again
                    cleaned_text = self._clean_text(p)
                    try:
                        para = Paragraph(cleaned_text, self.styles['Normal'])
                        elements.append(para)
                    except:
                        # If it still fails, add the text as a simple string
                        print(f"Warning: Could not parse paragraph: {cleaned_text[:50]}...")
                        elements.append(cleaned_text)
        
        return elements

    def _clean_text(self, text):
        # Remove any HTML-like tags
        text = re.sub('<[^<]+?>', '', text)
        # Replace special characters with their HTML entities
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # Remove any non-printable characters
        text = ''.join(char for char in text if ord(char) > 31 or ord(char) == 9)
        return text