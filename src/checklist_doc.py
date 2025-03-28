import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import numpy as np
from pathlib import Path
import threading
import signal
import sys
import time

from src.look_and_feel import error, success, warning, info, highlight
from src.print_pdf import PDFReportGenerator

class ChecklistDocumentor:
    def __init__(self, worker_erag_api, supervisor_erag_api=None):
        self.worker_erag_api = worker_erag_api
        self.supervisor_erag_api = supervisor_erag_api
        self.output_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
        os.makedirs(self.output_folder, exist_ok=True)
        self.file_path = None
        self.selected_sheet = None
        self.selected_column = None
        self.df = None
        self.sheets = []
        self.columns = []
        self.window = None
        self.preview_frame = None
        self.data_analysis = {}
        self.column_stats = {}
        self.use_supervisor = False
        self.supervisor_available = self.supervisor_erag_api is not None
        self.has_header = True  # Default to True since most files have headers
        self.output_file = None
        self.pdf_content = []
        self.paused = False
        self.setup_signal_handler()

    def setup_signal_handler(self):
        """Set up signal handler for Ctrl+C"""
        signal.signal(signal.SIGINT, self.handle_signal)

    def handle_signal(self, sig, frame):
        """Handle Ctrl+C by pausing execution"""
        if not self.paused:
            self.paused = True
            print(warning("\nScript paused. Press Enter to continue or Ctrl+C again to exit..."))
            try:
                user_input = input()
                self.paused = False
                print(info("Resuming execution..."))
            except KeyboardInterrupt:
                print(error("\nExiting script..."))
                sys.exit(0)
        else:
            print(error("\nExiting script..."))
            sys.exit(0)

    def check_if_paused(self):
        """Check if execution is paused and wait for Enter if needed"""
        while self.paused:
            time.sleep(0.1)  # Small sleep to prevent CPU hogging

    def analyze_file(self, file_path):
        """Analyze the loaded file and create statistics for sheets and columns."""
        try:
            file_stats = {
                "filename": os.path.basename(file_path),
                "file_size": os.path.getsize(file_path) / 1024,  # KB
                "file_type": Path(file_path).suffix,
                "sheets": [],
                "total_sheets": 0
            }
            
            if file_path.endswith('.csv'):
                # For CSV files
                df = pd.read_csv(file_path, header=0 if self.has_header else None)
                if not self.has_header:
                    df.columns = [f"Column_{i+1}" for i in range(len(df.columns))]
                
                self.sheets = ['main']  # CSV has only one sheet
                sheet_stats = self.analyze_dataframe(df)
                file_stats["sheets"].append({
                    "name": "main",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_stats": sheet_stats,
                    "potential_checklist_cols": self.identify_potential_checklist_cols(df)
                })
                file_stats["total_sheets"] = 1
                
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                # For Excel files
                excel_file = pd.ExcelFile(file_path)
                self.sheets = excel_file.sheet_names
                file_stats["total_sheets"] = len(self.sheets)
                
                for sheet in self.sheets:
                    df = pd.read_excel(file_path, sheet_name=sheet, header=0 if self.has_header else None)
                    if not self.has_header:
                        df.columns = [f"Column_{i+1}" for i in range(len(df.columns))]
                    
                    sheet_stats = self.analyze_dataframe(df)
                    file_stats["sheets"].append({
                        "name": sheet,
                        "rows": len(df),
                        "columns": len(df.columns),
                        "column_stats": sheet_stats,
                        "potential_checklist_cols": self.identify_potential_checklist_cols(df)
                    })
            
            return file_stats
        except Exception as e:
            error_message = f"Error analyzing file: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)
            return None

    def analyze_dataframe(self, df):
        """Analyze a dataframe and generate statistics for each column."""
        stats = {}
        for column in df.columns:
            col_stats = {
                "dtype": str(df[column].dtype),
                "unique_values": df[column].nunique(),
                "null_count": df[column].isna().sum(),
                "null_percentage": (df[column].isna().sum() / len(df)) * 100 if len(df) > 0 else 0,
                "sample_values": df[column].dropna().head(3).tolist(),
                "is_text": df[column].dtype == 'object',
                "avg_length": df[column].astype(str).str.len().mean() if df[column].dtype == 'object' else None
            }
            
            # Add statistics specific to numeric columns
            if np.issubdtype(df[column].dtype, np.number):
                col_stats.update({
                    "min": df[column].min() if not df[column].empty else None,
                    "max": df[column].max() if not df[column].empty else None,
                    "mean": df[column].mean() if not df[column].empty else None
                })
            
            stats[column] = col_stats
        
        return stats

    def identify_potential_checklist_cols(self, df):
        """Identify columns that might contain checklist items."""
        potential_cols = []
        
        for column in df.columns:
            # Skip columns with all NaN values
            if df[column].isna().all():
                continue
                
            # Good candidates are text columns with reasonable string lengths
            if df[column].dtype == 'object':
                # Check average string length (not too short, not too long)
                avg_len = df[column].astype(str).str.len().mean()
                if 10 <= avg_len <= 300:  # Increased upper limit for complex checklist items
                    score = 1
                    
                    # Look for columns with checklist-related names
                    col_lower = str(column).lower()
                    checklist_keywords = ['check', 'item', 'control', 'requirement', 'task', 'audit', 
                                         'compliance', 'procedure', 'standard', 'test', 'verification']
                    if any(keyword in col_lower for keyword in checklist_keywords):
                        score += 2
                    
                    # Look for columns with sentence-like content
                    sample = df[column].dropna().astype(str).head(10)
                    has_sentences = any(s.count(' ') >= 3 for s in sample)
                    if has_sentences:
                        score += 1
                    
                    # Check if multiple rows start with similar prefixes (like numbers or bullets)
                    prefixes = sample.str[:2].value_counts()
                    if (prefixes > 1).any():
                        score += 1
                    
                    potential_cols.append((column, score))
        
        # Sort by score descending
        potential_cols.sort(key=lambda x: x[1], reverse=True)
        return [col for col, score in potential_cols]

    def read_checklist_column(self, file_path, sheet_name, column_name):
        """Read the specified column from a CSV, XLS, or XLSX file and return the list of controls."""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=0 if self.has_header else None)
            if not self.has_header:
                df.columns = [f"Column_{i+1}" for i in range(len(df.columns))]
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=0 if self.has_header else None)
            if not self.has_header:
                df.columns = [f"Column_{i+1}" for i in range(len(df.columns))]
        else:
            raise ValueError("Unsupported file format. Please provide a CSV, XLS, or XLSX file.")
        
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the file.")
        
        # Extract the column, drop NaN, strip whitespace, and remove empty strings
        controls = [control.strip() for control in df[column_name].dropna().tolist() if control.strip() != ""]
        return controls, df

    def generate_worker_prompt(self, control):
        """Generate a detailed prompt for the worker AI model based on the audit control."""
        return f"""For the following specific audit control: '{control}', provide a detailed, PRACTICAL, and CONCRETE response. 
        
    Your response should be SPECIFICALLY TAILORED to this exact control, not generic audit guidance. Focus on how THIS PARTICULAR control would be tested, documented, and evaluated in a real-world environment.

    1. **Control Interpretation:** 
    - Provide a specific interpretation of what THIS control means in practical terms
    - What specific risks or requirements does THIS control address?
    - What specific business processes or systems does THIS control apply to?

    2. **How to Test THIS SPECIFIC Control:** 
    - What SPECIFIC preparatory steps are needed to test THIS control (exact documents, systems, or tools needed)
    - What REAL sampling approach makes sense for THIS control (exact sample sizes, selection criteria)
    - Provide DETAILED test steps that an auditor could follow verbatim to test THIS control
    - List SPECIFIC validation methods applicable to THIS control
    - Name ACTUAL tools, reports, or resources needed for THIS control test

    3. **How to Document THIS SPECIFIC Control:**
    - What EXACT documentation formats are required for THIS control?
    - What SPECIFIC screenshots or evidence should be collected for THIS control?
    - What ACTUAL templates should be used for THIS control?
    - How should documentation be organized for THIS control?
    - What approval workflows apply to THIS control?

    4. **Expected Findings for THIS SPECIFIC Control:** Use the exact format "It is an audit finding if...":
    - Describe what ACTUAL compliant implementation of THIS control looks like
    
    - Critical Findings for THIS CONTROL:
        * It is a critical audit finding if... [specific scenario for THIS control]
        * It is a critical audit finding if... [specific scenario for THIS control]
        * It is a critical audit finding if... [specific scenario for THIS control]
    
    - Major Findings for THIS CONTROL:
        * It is a major audit finding if... [specific scenario for THIS control]
        * It is a major audit finding if... [specific scenario for THIS control]
        * It is a major audit finding if... [specific scenario for THIS control]
    
    - Minor Findings for THIS CONTROL:
        * It is a minor audit finding if... [specific scenario for THIS control]
        * It is a minor audit finding if... [specific scenario for THIS control]
        * It is a minor audit finding if... [specific scenario for THIS control]

    5. **Risk Assessment for THIS SPECIFIC Control:**
    - What SPECIFIC risks materialize if THIS control fails? (Be concrete)
    - What ACTUAL impact (operational, financial, reputational) occurs if THIS control fails?
    - How does failure of THIS control affect regulatory compliance?

    6. **Regulatory Alignment for THIS SPECIFIC Control:**
    - What SPECIFIC regulations or standards does THIS control help satisfy?
    - Cite ACTUAL regulatory requirements, with specific references where possible
    - What CONCRETE compliance outcomes does THIS control support?

    7. **Detailed Documentation Example for THIS SPECIFIC Control:**
    - Provide a REALISTIC, DETAILED example that directly relates to THIS control
    - Use CONCRETE examples, realistic dates, sample sizes, and findings
    - Include ACTUAL test steps an auditor would perform for THIS control
    - Show REAL evidence that would be collected for THIS control
    - Create SPECIFIC, NOT GENERIC test results for THIS control

    8. **Tips and Tricks for THIS SPECIFIC Control:**
    - Share expert techniques SPECIFICALLY for testing THIS control
    - What common mistakes happen when testing THIS PARTICULAR control?
    - What would experienced auditors focus on for THIS SPECIFIC control?
    - Provide time-saving approaches SPECIFIC to THIS control
    - How should auditors handle objections SPECIFICALLY about THIS control?

    IMPORTANT: Avoid generic audit language. Every response should be specifically tailored to the exact control provided. Use concrete examples, specific systems, realistic documents, and practical test steps that directly relate to this control.

    Ensure your response is highly detailed, practical, and actionable - as if writing for a junior auditor who needs specific guidance on how to test THIS PARTICULAR control."""

    def generate_supervisor_prompt(self, control, worker_response):
        """Generate a prompt for the supervisor AI model to enhance the worker's response."""
        return f"""You are a senior audit supervisor reviewing documentation for the following specific control: '{control}'

    Below is the initial documentation prepared by a junior auditor:

    {worker_response}

    Your task is to review, enhance, and make this documentation more SPECIFIC and PRACTICAL for THIS PARTICULAR control. The current documentation may be too generic and theoretical - your job is to make it concrete and directly applicable to this specific control.

    1. **Make the Control Interpretation More Specific:**
    - Ensure the interpretation directly addresses THIS control's practical implementation
    - Add specific business context about where and how THIS control operates
    - Remove any generic audit language that could apply to any control

    2. **Make Test Procedures Concrete and Practical:**
    - Replace any generic test steps with SPECIFIC procedures for THIS control
    - Add realistic sample sizes, specific documents to review, and exact systems to check
    - Ensure test procedures are detailed enough that a junior auditor could follow them exactly

    3. **Make Documentation Guidance Control-Specific:**
    - Replace generic documentation advice with SPECIFIC guidance for THIS control
    - Name actual documents, screenshots, and evidence needed for THIS control
    - Provide realistic examples of documentation specific to THIS control

    4. **Make Finding Scenarios Control-Specific:**
    - Ensure all "It is a [severity] audit finding if..." statements directly relate to THIS control
    - Replace any generic findings with scenarios that specifically relate to THIS control
    - Add concrete examples that show deep understanding of how THIS control could fail

    5. **Make Risk Assessment Control-Specific:**
    - Replace generic risks with actual consequences specific to THIS control's failure
    - Add industry-specific impacts that directly relate to THIS control
    - Provide quantified or measurable impacts where possible

    6. **Add Specific Regulatory Citations:**
    - Replace general regulatory mentions with specific sections/paragraphs
    - Add actual citation numbers or specific requirements from standards
    - Connect THIS control directly to specific compliance requirements

    7. **Make the Example Documentation Realistic:**
    - Ensure the example reads like a REAL audit report for THIS control
    - Add realistic dates, sample sizes, specific findings, and detailed test steps
    - Make sure all parts of the example are specific to THIS control

    8. **Add Control-Specific Expert Insights:**
    - Share specialized knowledge that only applies to testing THIS control
    - Add practical shortcuts or efficiencies specific to THIS control
    - Provide real-world challenges unique to THIS control

    Your response must be concrete, practical, and SPECIFICALLY tailored to this exact control. Remove any content that reads like generic audit guidance. Your goal is to transform theoretical audit concepts into practical, actionable guidance that a junior auditor could follow to test THIS SPECIFIC control effectively."""

    def write_to_output_file(self, content):
        """Write content to the output file and flush immediately."""
        if self.output_file:
            try:
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(content)
                    f.flush()  # Ensure content is written immediately
                return True
            except Exception as e:
                print(error(f"Error writing to output file: {str(e)}"))
                return False
        return False

    def process_control_with_worker(self, control):
        """Process a single control using the worker AI model."""
        prompt = self.generate_worker_prompt(control)
        worker_response = self.worker_erag_api.chat([
            {"role": "system", "content": "You are an expert audit specialist providing detailed documentation for audit controls."},
            {"role": "user", "content": prompt}
        ])
        return worker_response

    def process_control_with_supervisor(self, control, worker_response):
        """Process a control with supervisor review of the worker's response."""
        prompt = self.generate_supervisor_prompt(control, worker_response)
        supervisor_response = self.supervisor_erag_api.chat([
            {"role": "system", "content": "You are a senior audit expert reviewing and enhancing audit control documentation."},
            {"role": "user", "content": prompt}
        ])
        return supervisor_response

    def process_checklist(self):
        """Process the checklist and generate responses for each control."""
        try:
            if not self.file_path or not self.selected_column:
                messagebox.showerror("Error", "Please select a file and column first.")
                return None
                
            controls, _ = self.read_checklist_column(self.file_path, self.selected_sheet, self.selected_column)
            print(info(f"Found {len(controls)} controls in column '{self.selected_column}'."))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = os.path.join(self.output_folder, f"checklist_documentation_{timestamp}.txt")
            pdf_file = os.path.join(self.output_folder, f"checklist_documentation_{timestamp}.pdf")
            
            # Configuration info for confirmation dialog
            config_info = (
                f"Configuration:\n"
                f"- Using worker model: {self.worker_erag_api.model}\n"
            )
            
            if self.use_supervisor and self.supervisor_available:
                config_info += f"- Using supervisor model: {self.supervisor_erag_api.model}\n"
                config_info += "- Two-phase processing enabled (enhanced quality)\n"
            else:
                config_info += "- Supervisor review disabled\n"
            
            # Ask for confirmation with progress details
            confirm = messagebox.askyesno(
                "Confirm Processing", 
                f"Ready to process {len(controls)} controls from '{self.selected_column}' column.\n\n"
                f"{config_info}\n"
                f"This may take some time depending on the number of controls.\n"
                f"Results will be saved to:\n"
                f"- Text file: {self.output_file}\n"
                f"- PDF report: {pdf_file}\n\n"
                f"Press Ctrl+C during processing to pause/resume.\n\n"
                f"Do you want to continue?"
            )
            
            if not confirm:
                return None
            
            progress_window = self.create_progress_window(len(controls))
            progress_var = progress_window.progress_var
            progress_label = progress_window.progress_label
            progress_window.window.update()
            
            # Initialize the output file with header
            file_header = (
                f"Audit Control Documentation\n"
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"File: {os.path.basename(self.file_path)}\n"
                f"Sheet: {self.selected_sheet}\n"
                f"Column: {self.selected_column}\n"
                f"Models used: {self.worker_erag_api.model}"
                f"{' + ' + self.supervisor_erag_api.model if self.use_supervisor and self.supervisor_available else ''}\n\n"
                f"{'='*80}\n\n"
            )
            self.write_to_output_file(file_header)
            
            # Initialize PDF content list
            self.pdf_content = []
            findings = []
            
            for i, control in enumerate(controls, 1):
                # Check if paused
                self.check_if_paused()
                
                # Update progress
                progress_var.set((i - 0.5) / len(controls) * 100)  # Halfway through each control
                progress_label.config(text=f"Processing control {i}/{len(controls)}...")
                progress_window.window.update()
                
                print(highlight(f"Processing control {i}/{len(controls)}: {control}"))
                
                # First, process with worker
                worker_response = self.process_control_with_worker(control)
                
                final_response = worker_response
                
                # If supervisor is enabled and available, use it to improve the response
                if self.use_supervisor and self.supervisor_available:
                    progress_label.config(text=f"Supervisor reviewing control {i}/{len(controls)}...")
                    progress_window.window.update()
                    
                    # Check if paused before supervisor processing
                    self.check_if_paused()
                    
                    supervisor_response = self.process_control_with_supervisor(control, worker_response)
                    final_response = supervisor_response
                
                # Format the result for text output
                text_output = f"Control {i}: {control}\n\n{final_response}\n\n{'='*80}\n\n"
                
                # Write this control's documentation to the output file
                self.write_to_output_file(text_output)
                
                # Format content for PDF with proper markdown structure
                pdf_content = self.format_control_for_pdf(control, final_response, i)
                
                # Add to PDF content
                self.pdf_content.append((f"Control {i}: {control}", [], pdf_content))
                
                # Extract key findings for the PDF report
                findings.append(f"Control {i}: {control}")
                
                # Update progress to completion for this control
                progress_var.set(i / len(controls) * 100)
                progress_window.window.update()
                
                # Report progress
                progress_pct = (i / len(controls)) * 100
                print(info(f"Progress: {progress_pct:.1f}% - {i}/{len(controls)} controls processed"))
            
            # Close progress window
            progress_window.window.destroy()
            
            # Generate PDF report
            self.generate_pdf_report(findings, pdf_file)
            
            print(success(f"Checklist documentation completed successfully."))
            print(success(f"Text results saved to: {self.output_file}"))
            print(success(f"PDF report saved to: {pdf_file}"))
            
            messagebox.showinfo("Success", 
                                f"Checklist documentation completed successfully.\n\n"
                                f"Results saved to:\n"
                                f"- Text file: {self.output_file}\n"
                                f"- PDF report: {pdf_file}")
            
            return self.output_file
        
        except Exception as e:
            error_message = f"An error occurred while processing the checklist: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)
            return None

    def generate_pdf_report(self, findings, pdf_file):
        """Generate a PDF report from the processed controls."""
        try:
            # Extract base filename without extension
            base_filename = os.path.splitext(os.path.basename(pdf_file))[0]
            
            # Create PDF generator
            model_info = self.worker_erag_api.model
            if self.use_supervisor and self.supervisor_available:
                model_info += f" + {self.supervisor_erag_api.model}"
            
            pdf_generator = PDFReportGenerator(
                self.output_folder, 
                model_info,
                os.path.basename(self.file_path)
            )
            
            # Generate the PDF report
            report_title = f"Audit Control Documentation - {os.path.basename(self.file_path)}"
            pdf_file = pdf_generator.create_enhanced_pdf_report(
                findings,
                self.pdf_content,
                [],  # No images for now
                filename=base_filename,
                report_title=report_title
            )
            
            return pdf_file
        except Exception as e:
            error_message = f"Error generating PDF report: {str(e)}"
            print(error(error_message))
            return None
        
    def format_control_for_pdf(self, control, response, index):
        """Format control documentation for PDF with proper markdown structure."""
        # Extract title if available (often starts with "## Audit Control: ")
        lines = response.split('\n')
        title_line = next((line for line in lines if line.startswith("## Audit Control:") or line.startswith("## ")), None)
        
        # If no title found, create one
        if not title_line:
            title = f"Audit Control: {control}"
        else:
            title = title_line.strip("# ")
        
        # Check if response already has markdown structure
        has_markdown_headings = any(line.startswith('#') for line in lines)
        
        if has_markdown_headings:
            # Response already has markdown structure, just ensure proper overall structure
            formatted_content = f"""# Control {index}: {control}

    {response}
    """
        else:
            # Response lacks markdown structure, add it
            sections = [
                "Control Interpretation",
                "How to Test",
                "How to Document",
                "Expected Findings",
                "Risk Assessment",
                "Regulatory Alignment",
                "Detailed Documentation Example",
                "Tips and Tricks"
            ]
            
            # Find each section in the response
            formatted_sections = []
            formatted_sections.append(f"# Control {index}: {control}")
            formatted_sections.append(f"## {title}")
            
            current_text = []
            current_section = None
            
            for line in lines:
                # Check if this line starts a new section
                found_section = False
                for section in sections:
                    if line.strip().startswith(section + ":") or line.strip() == section:
                        # End previous section if any
                        if current_section and current_text:
                            formatted_sections.append(f"### {current_section}")
                            formatted_sections.append('\n'.join(current_text))
                            current_text = []
                        
                        # Start new section
                        current_section = section
                        found_section = True
                        break
                
                if found_section:
                    continue
                
                # Add line to current section
                if current_section:
                    current_text.append(line)
                else:
                    # If no section has been found yet, this is introductory text
                    current_text.append(line)
            
            # Add final section
            if current_section and current_text:
                formatted_sections.append(f"### {current_section}")
                formatted_sections.append('\n'.join(current_text))
            elif current_text:
                # Any remaining text without a section
                formatted_sections.append('\n'.join(current_text))
            
            formatted_content = '\n\n'.join(formatted_sections)
        
        return formatted_content

    def create_progress_window(self, total_items):
        """Create a progress window for showing processing status."""
        window = tk.Toplevel()
        window.title("Processing Checklist")
        window.geometry("500x220")
        window.resizable(False, False)
        
        frame = ttk.Frame(window, padding="20")
        frame.pack(fill="both", expand=True)
        
        # Configuration display
        config_text = f"Worker Model: {self.worker_erag_api.model}"
        if self.use_supervisor and self.supervisor_available:
            config_text += f"\nSupervisor Model: {self.supervisor_erag_api.model}"
        
        ttk.Label(frame, text="Documenting Checklist Items", font=("Arial", 12, "bold")).pack(pady=(0, 5))
        ttk.Label(frame, text=config_text, foreground="blue").pack(pady=(0, 15))
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100, length=400)
        progress_bar.pack(fill="x", pady=5)
        
        progress_label = ttk.Label(frame, text="Initializing...")
        progress_label.pack(pady=5)
        
        # Pause instructions
        ttk.Label(
            frame, 
            text="Press Ctrl+C in the console window to pause/resume processing.",
            foreground="dark green",
            font=("Arial", 9)
        ).pack(pady=(10, 5))
        
        # Cancel button
        cancel_button = ttk.Button(frame, text="Cancel", command=lambda: None)  # Placeholder
        cancel_button.pack(pady=5)
        
        # Center the window
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')
        
        # Make it modal
        window.transient(window.master)
        window.grab_set()
        
        # Create a container object to hold references
        class ProgressWindow:
            def __init__(self, window, progress_var, progress_label, cancel_button):
                self.window = window
                self.progress_var = progress_var
                self.progress_label = progress_label
                self.cancel_button = cancel_button
        
        return ProgressWindow(window, progress_var, progress_label, cancel_button)

    def create_file_selector(self):
        """Create a file selector dialog and analyze the selected file."""
        file_path = filedialog.askopenfilename(
            title="Select Checklist File",
            filetypes=[
                ("Excel files", "*.xlsx;*.xls"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return None
            
        self.file_path = file_path
        self.file_label.config(text=os.path.basename(file_path))
        print(info(f"Analyzing file: {os.path.basename(file_path)}"))
        
        # Show a loading indicator
        loading_window = tk.Toplevel(self.window)
        loading_window.title("Loading")
        loading_window.geometry("300x100")
        ttk.Label(loading_window, text="Analyzing file structure...", padding=20).pack()
        progress = ttk.Progressbar(loading_window, mode="indeterminate")
        progress.pack(fill="x", padx=20, pady=10)
        progress.start()
        
        # Center the loading window
        loading_window.update_idletasks()
        loading_window.geometry(f'+{self.window.winfo_x() + 100}+{self.window.winfo_y() + 100}')
        loading_window.transient(self.window)
        loading_window.grab_set()
        loading_window.update()
        
        try:
            # Analyze the file in a separate thread
            def analyze_thread():
                self.data_analysis = self.analyze_file(file_path)
                loading_window.destroy()
                
                if self.data_analysis:
                    # Update UI with analysis results
                    self.window.after(0, self.update_ui_with_analysis)
            
            thread = threading.Thread(target=analyze_thread)
            thread.daemon = True
            thread.start()
            
            return self.data_analysis
            
        except Exception as e:
            loading_window.destroy()
            error_message = f"Error analyzing file: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)
            return None

    def update_ui_with_analysis(self):
        """Update the UI with the results of file analysis."""
        if not self.data_analysis:
            return
            
        # Clear previous content
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
            
        # File info section
        file_info_frame = ttk.LabelFrame(self.preview_frame, text="File Information")
        file_info_frame.pack(fill="x", padx=10, pady=5)
        
        file_info = (
            f"Filename: {self.data_analysis['filename']}\n"
            f"File Type: {self.data_analysis['file_type']}\n"
            f"File Size: {self.data_analysis['file_size']:.2f} KB\n"
            f"Total Sheets: {self.data_analysis['total_sheets']}"
        )
        ttk.Label(file_info_frame, text=file_info, justify="left", padding=5).pack(anchor="w")
        
        # Sheet selection section (for Excel)
        if self.data_analysis['file_type'] in ['.xlsx', '.xls']:
            sheet_frame = ttk.LabelFrame(self.preview_frame, text="Step 2: Select Sheet")
            sheet_frame.pack(fill="x", padx=10, pady=5)
            
            sheet_list = ttk.Treeview(sheet_frame, columns=("rows", "columns", "potential"), show="headings", height=min(5, len(self.data_analysis["sheets"])))
            sheet_list.heading("rows", text="Rows")
            sheet_list.heading("columns", text="Columns")
            sheet_list.heading("potential", text="Potential Checklist Columns")
            sheet_list.column("rows", width=80)
            sheet_list.column("columns", width=80)
            sheet_list.column("potential", width=200)
            
            for sheet_info in self.data_analysis["sheets"]:
                potential_cols = ", ".join(sheet_info["potential_checklist_cols"][:3]) if sheet_info["potential_checklist_cols"] else "None found"
                sheet_list.insert("", "end", text=sheet_info["name"], values=(
                    sheet_info["rows"],
                    sheet_info["columns"],
                    potential_cols
                ), tags=(sheet_info["name"],))
                
            sheet_list.pack(fill="x", padx=5, pady=5)
            
            # Sheet selection button
            def on_select_sheet():
                selected_items = sheet_list.selection()
                if not selected_items:
                    messagebox.showwarning("Warning", "Please select a sheet first.")
                    return
                
                sheet_name = sheet_list.item(selected_items[0], "tags")[0]
                self.select_sheet(sheet_name)
                
            ttk.Button(sheet_frame, text="Select Sheet", command=on_select_sheet).pack(pady=5)
            
        else:
            # For CSV, automatically select the 'main' sheet
            self.select_sheet("main")

    def select_sheet(self, sheet_name):
        """Handle sheet selection and show column selection UI."""
        self.selected_sheet = sheet_name
        print(info(f"Selected sheet: {sheet_name}"))
        
        # Find the sheet info
        sheet_info = next((s for s in self.data_analysis["sheets"] if s["name"] == sheet_name), None)
        if not sheet_info:
            return
            
        # Load the dataframe for this sheet
        if self.data_analysis['file_type'] == '.csv':
            self.df = pd.read_csv(self.file_path, header=0 if self.has_header else None)
            if not self.has_header:
                self.df.columns = [f"Column_{i+1}" for i in range(len(self.df.columns))]
        else:
            self.df = pd.read_excel(self.file_path, sheet_name=sheet_name, header=0 if self.has_header else None)
            if not self.has_header:
                self.df.columns = [f"Column_{i+1}" for i in range(len(self.df.columns))]
            
        # Store column stats for this sheet
        self.column_stats = sheet_info["column_stats"]
        
        # Update UI with column selection
        self.show_column_selection(sheet_info)

    def show_column_selection(self, sheet_info):
        """Show the column selection UI based on sheet analysis."""
        # Clear previous column selection UI if exists
        for widget in self.preview_frame.winfo_children():
            if hasattr(widget, 'tag') and widget.tag == 'column_selection':
                widget.destroy()
                
        # Create column selection frame
        column_frame = ttk.LabelFrame(self.preview_frame, text="Step 3: Select Column with Checklist Items")
        column_frame.tag = 'column_selection'  # Custom attribute to identify this frame
        column_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create a treeview with column statistics
        columns_list = ttk.Treeview(
            column_frame, 
            columns=("type", "unique", "nulls", "samples"), 
            show="headings", 
            height=min(10, len(self.column_stats))
        )
        columns_list.heading("type", text="Type")
        columns_list.heading("unique", text="Unique Values")
        columns_list.heading("nulls", text="Nulls %")
        columns_list.heading("samples", text="Sample Values")
        
        columns_list.column("type", width=80)
        columns_list.column("unique", width=100)
        columns_list.column("nulls", width=80)
        columns_list.column("samples", width=250)
        
        # Add column info to treeview
        for column_name, stats in self.column_stats.items():
            sample_text = ", ".join([str(s) for s in stats["sample_values"][:2]]) if stats["sample_values"] else "No samples"
            if len(sample_text) > 40:
                sample_text = sample_text[:37] + "..."
                
            columns_list.insert("", "end", text=column_name, values=(
                stats["dtype"],
                stats["unique_values"],
                f"{stats['null_percentage']:.1f}%",
                sample_text
            ), tags=(column_name,))
        
        columns_list.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Highlight recommended columns
        potential_cols = sheet_info["potential_checklist_cols"]
        for i, item_id in enumerate(columns_list.get_children()):
            column_name = columns_list.item(item_id, "tags")[0]
            if column_name in potential_cols:
                columns_list.item(item_id, tags=(column_name, "recommended"))
                columns_list.tag_configure("recommended", background="#e6f3ff")
        
        # Add recommendation label if available
        if potential_cols:
            ttk.Label(
                column_frame, 
                text=f"Recommended columns (highlighted): {', '.join(potential_cols[:3])}" + 
                     (" ..." if len(potential_cols) > 3 else ""),
                foreground="blue"
            ).pack(anchor="w", padx=5)
        
        # Column selection
        def on_column_select(event):
            selected_items = columns_list.selection()
            if not selected_items:
                return
                
            column_name = columns_list.item(selected_items[0], "tags")[0]
            self.preview_column(column_name)
            
        columns_list.bind("<<TreeviewSelect>>", on_column_select)
        
        # Create preview area
        preview_label = ttk.Label(column_frame, text="Column Preview:", anchor="w")
        preview_label.pack(fill="x", padx=5, pady=(10, 0))
        
        preview_text = tk.Text(column_frame, height=10, width=60, wrap="word")
        preview_text.pack(fill="both", padx=5, pady=5)
        
        # Set as instance variable to access in preview_column
        self.preview_text = preview_text
        
        # Create configuration frame for options
        config_frame = ttk.LabelFrame(column_frame, text="Processing Configuration")
        config_frame.pack(fill="x", padx=5, pady=5)
        
        # Header checkbox
        self.has_header_var = tk.BooleanVar(value=self.has_header)
        header_check = ttk.Checkbutton(
            config_frame, 
            text="File has header row (first row contains column names)", 
            variable=self.has_header_var
        )
        header_check.pack(anchor="w", padx=5, pady=5)
        
        # Update the has_header flag when checkbox changes
        def update_header_setting(*args):
            old_value = self.has_header
            self.has_header = self.has_header_var.get()
            if old_value != self.has_header:
                # If the setting changed, reload the file with the new setting
                messagebox.showinfo("Header Setting Changed", 
                                  "The file will be reloaded with the new header setting.")
                self.create_file_selector()
        
        self.has_header_var.trace_add("write", update_header_setting)
        
        # Supervisor checkbox (if available)
        if self.supervisor_available:
            self.use_supervisor_var = tk.BooleanVar(value=True)
            supervisor_check = ttk.Checkbutton(
                config_frame, 
                text=f"Use supervisor model ({self.supervisor_erag_api.model}) to enhance results", 
                variable=self.use_supervisor_var
            )
            supervisor_check.pack(anchor="w", padx=5, pady=5)
            
            ttk.Label(
                config_frame,
                text="Using the supervisor model will produce more comprehensive and higher quality\n"
                     "documentation, but will take approximately twice as long to process.",
                foreground="gray"
            ).pack(anchor="w", padx=5, pady=0)
            
            # Update the use_supervisor flag when checkbox changes
            def update_supervisor_setting(*args):
                self.use_supervisor = self.use_supervisor_var.get()
            
            self.use_supervisor_var.trace_add("write", update_supervisor_setting)
            self.use_supervisor = self.use_supervisor_var.get()
        
        # Output format checkbox
        output_frame = ttk.Frame(config_frame)
        output_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(output_frame, text="Output Formats:").pack(anchor="w")
        
        self.output_txt_var = tk.BooleanVar(value=True)
        txt_check = ttk.Checkbutton(
            output_frame, 
            text="Text File (.txt)", 
            variable=self.output_txt_var
        )
        txt_check.pack(anchor="w", padx=20)
        
        self.output_pdf_var = tk.BooleanVar(value=True)
        pdf_check = ttk.Checkbutton(
            output_frame, 
            text="PDF Report (.pdf)", 
            variable=self.output_pdf_var
        )
        pdf_check.pack(anchor="w", padx=20)
        
        # Button to select column and proceed
        def on_select_column():
            selected_items = columns_list.selection()
            if not selected_items:
                messagebox.showwarning("Warning", "Please select a column first.")
                return
                
            column_name = columns_list.item(selected_items[0], "tags")[0]
            self.selected_column = column_name
            print(success(f"Selected column: {column_name}"))
            
            # Show confirmation and proceed
            if not messagebox.askyesno("Confirm Selection", 
                                     f"You selected column: {column_name}\n\n"
                                     f"Do you want to proceed with documenting this column?"):
                return
            
            # Check if at least one output format is selected
            if not self.output_txt_var.get() and not self.output_pdf_var.get():
                messagebox.showwarning("Warning", "Please select at least one output format (Text or PDF).")
                return
                
            # Close the window and process
            self.window.destroy()
            self.process_checklist()
            
        ttk.Button(column_frame, text="Document Selected Column", command=on_select_column).pack(pady=10)

    def preview_column(self, column_name):
        """Preview a selected column's contents."""
        if self.df is None or column_name not in self.df.columns:
            return
            
        # Clear previous preview
        self.preview_text.delete(1.0, tk.END)
        
        # Get data for the column
        column_data = self.df[column_name].dropna()
        stats = self.column_stats[column_name]
        
        # Add header information
        self.preview_text.insert(tk.END, f"Column: {column_name}\n")
        self.preview_text.insert(tk.END, f"Data Type: {stats['dtype']}\n")
        self.preview_text.insert(tk.END, f"Unique Values: {stats['unique_values']}\n")
        self.preview_text.insert(tk.END, f"Total Records: {len(column_data)}\n\n")
        
        # Add preview of values
        self.preview_text.insert(tk.END, "Preview of values:\n")
        for i, value in enumerate(column_data.head(10), 1):
            self.preview_text.insert(tk.END, f"{i}. {value}\n")
            
        if len(column_data) > 10:
            self.preview_text.insert(tk.END, f"... and {len(column_data) - 10} more records\n")

    def create_ui_window(self):
        """Create the main UI window for the checklist documenter."""
        window = tk.Toplevel()
        window.title("Checklist Documenter")
        window.geometry("800x700")
        window.minsize(700, 500)
        
        # Store reference to window
        self.window = window
        
        # Main frame
        main_frame = ttk.Frame(window, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(
            header_frame, 
            text="Audit Checklist Documenter", 
            font=("Arial", 14, "bold")
        ).pack(side="left")
        
        if self.supervisor_available:
            model_info = f"Using: {self.worker_erag_api.model} | Supervisor: {self.supervisor_erag_api.model}"
        else:
            model_info = f"Using: {self.worker_erag_api.model}"
            
        ttk.Label(
            header_frame,
            text=model_info,
            font=("Arial", 9),
            foreground="blue"
        ).pack(side="right")
        
        description = ttk.Label(
            main_frame,
            text="This tool helps you create comprehensive audit documentation for checklist items.\n"
                 "Select an Excel or CSV file containing your audit controls or checklist items.\n"
                 "Press Ctrl+C during processing to pause/resume.",
            justify="left",
            wraplength=780
        )
        description.pack(fill="x", pady=(0, 10))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="Step 1: Select Checklist File")
        file_frame.pack(fill="x", padx=5, pady=5)
        
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.pack(side="left", padx=5, pady=5, fill="x", expand=True)
        
        browse_button = ttk.Button(file_frame, text="Browse...", command=self.create_file_selector)
        browse_button.pack(side="right", padx=5, pady=5)
        
        # Preview frame - will be populated after file selection
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Store reference to preview frame
        self.preview_frame = preview_frame
        
        # Button to cancel
        ttk.Button(main_frame, text="Cancel", command=window.destroy).pack(side="right", padx=5, pady=5)
        
        # Center the window on the screen
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        x = (window.winfo_screenwidth() // 2) - (width // 2)
        y = (window.winfo_screenheight() // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')
        
        # Make the window modal
        window.transient(window.master)
        window.grab_set()
        
        return window

    def run(self):
        """Run the checklist documenter with a graphical interface."""
        try:
            print(info("Starting Checklist Documenter..."))
            
            if self.supervisor_available:
                print(info(f"Worker model: {self.worker_erag_api.model}"))
                print(info(f"Supervisor model: {self.supervisor_erag_api.model}"))
            else:
                print(info(f"Using model: {self.worker_erag_api.model}"))
                print(info("Supervisor model not available - operating in single model mode"))
                
            print(info("Press Ctrl+C during processing to pause/resume."))
                
            window = self.create_ui_window()
            window.wait_window()  # Wait for the window to close
            
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error(error_message))
            messagebox.showerror("Error", error_message)