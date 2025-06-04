# src/processors/pdf_processor.py

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import PyPDF2
import pdfplumber
import re

class PDFProcessor:
    """Process PDF files from portfolio"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.processed_content = []

    def process_all_pdfs(self) -> List[Dict[str, Any]]:
        """Process all PDF files in the export and raw data folder"""
        print("🔍 Finding PDF files...")

        # Find PDFs in the notion export folder
        notion_pdf_files = list(self.base_path.rglob("*.pdf"))

        # Also look for PDFs in the parent raw folder (like mingresume.pdf)
        raw_folder = self.base_path.parent if self.base_path.name == "notion_export" else self.base_path
        raw_pdf_files = list(raw_folder.glob("*.pdf"))

        # Combine and deduplicate
        all_pdf_files = list(set(notion_pdf_files + raw_pdf_files))

        print(f"Found {len(all_pdf_files)} PDF files:")
        for pdf_file in all_pdf_files:
            relative_path = pdf_file.relative_to(raw_folder.parent) if raw_folder.parent in pdf_file.parents else pdf_file.name
            print(f"  - {relative_path}")

        for pdf_file in all_pdf_files:
            try:
                content = self.process_single_pdf(pdf_file)
                if content:
                    self.processed_content.append(content)
                    print(f"✅ Processed: {pdf_file.name}")
            except Exception as e:
                print(f"❌ Error processing {pdf_file.name}: {e}")

        return self.processed_content

    def process_single_pdf(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Process a single PDF file"""
        # Try pdfplumber first (better for complex layouts)
        text_content = self._extract_with_pdfplumber(file_path)

        # Fallback to PyPDF2 if pdfplumber fails
        if not text_content or len(text_content.strip()) < 50:
            text_content = self._extract_with_pypdf2(file_path)

        # Skip if no meaningful content extracted
        if not text_content or len(text_content.strip()) < 50:
            print(f"⚠️  Minimal text extracted from {file_path.name}")
            return None

        # Process and clean content
        cleaned_content = self._clean_content(text_content)

        # Extract metadata
        metadata = self._extract_metadata(file_path, cleaned_content)

        # Extract sections/pages
        sections = self._extract_sections(cleaned_content, file_path)

        return {
            'id': self._generate_id(file_path),
            'source_file': str(file_path),
            'type': 'pdf',
            'title': metadata['title'],
            'category': metadata['category'],
            'raw_content': text_content,
            'cleaned_content': cleaned_content,
            'sections': sections,
            'metadata': metadata,
            'word_count': len(cleaned_content.split()),
            'page_count': metadata.get('page_count', 0)
        }

    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """Extract text using pdfplumber (better for tables/complex layouts)"""
        try:
            text_content = ""
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num} ---\n"
                        text_content += page_text + "\n"

            return text_content
        except Exception as e:
            print(f"pdfplumber failed for {file_path.name}: {e}")
            return ""

    def _extract_with_pypdf2(self, file_path: Path) -> str:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            text_content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content += f"\n--- Page {page_num} ---\n"
                        text_content += page_text + "\n"

            return text_content
        except Exception as e:
            print(f"PyPDF2 failed for {file_path.name}: {e}")
            return ""

    def _clean_content(self, content: str) -> str:
        """Clean extracted PDF content"""
        # Remove excessive whitespace and formatting artifacts
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Multiple newlines
        content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces/tabs

        # Remove common PDF artifacts
        content = re.sub(r'---\s*Page\s+\d+\s*---', '\n[PAGE BREAK]\n', content)

        # Fix common OCR/extraction issues
        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)  # Missing spaces
        content = re.sub(r'(\d+)([A-Za-z])', r'\1 \2', content)  # Number-letter joins

        # Remove header/footer repetitions (simple heuristic)
        lines = content.split('\n')
        cleaned_lines = []
        prev_line = ""

        for line in lines:
            line = line.strip()
            # Skip very short repeated lines (likely headers/footers)
            if len(line) > 3 and line != prev_line:
                cleaned_lines.append(line)
            prev_line = line

        return '\n'.join(cleaned_lines).strip()

    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from PDF file and content"""
        file_name = file_path.stem

        # Categorize based on filename patterns
        category = self._categorize_pdf(file_name, content)
        title = self._extract_title(file_name, content)

        # Try to get page count
        page_count = content.count('[PAGE BREAK]') + 1 if '[PAGE BREAK]' in content else 1

        # Extract project association if in project folder
        project_name = None
        if 'Projects' in str(file_path):
            project_parts = str(file_path).split('/')
            for part in project_parts:
                if part.startswith('Projects') and len(part) > 8:
                    continue
                elif any(keyword in part.lower() for keyword in ['cpf', 'evam', 'skin', 'nlp', 'expo', 'ura', 'sutd', 'nothing']):
                    project_name = part
                    break

        # Get relative path safely
        try:
            relative_path = str(file_path.relative_to(self.base_path))
        except ValueError:
            # File is outside base_path (like mingresume.pdf)
            relative_path = f"raw/{file_path.name}"

        return {
            'title': title,
            'category': category,
            'file_name': file_name,
            'relative_path': relative_path,
            'page_count': page_count,
            'project_name': project_name,
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'last_modified': file_path.stat().st_mtime if file_path.exists() else None
        }

    def _categorize_pdf(self, file_name: str, content: str) -> str:
        """Categorize PDF based on filename and content"""
        file_lower = file_name.lower()
        content_lower = content.lower()

        # Check for resume/CV first (common patterns)
        if any(keyword in file_lower for keyword in ['resume', 'cv', 'ming']):
            return 'resume'
        elif any(keyword in file_lower for keyword in ['final_report', 'report', 'final_presentation']):
            return 'project_report'
        elif any(keyword in file_lower for keyword in ['presentation', 'slide']):
            return 'presentation'
        elif any(keyword in file_lower for keyword in ['proposal', 'plan']):
            return 'proposal'
        elif 'team' in file_lower:
            return 'team_document'
        else:
            # Try to infer from content
            if any(keyword in content_lower for keyword in ['experience', 'education', 'skills', 'objective']):
                return 'resume'
            elif any(keyword in content_lower for keyword in ['abstract', 'introduction', 'methodology', 'conclusion']):
                return 'academic_paper'
            elif any(keyword in content_lower for keyword in ['agenda', 'meeting', 'minutes']):
                return 'meeting_document'
            else:
                return 'general_document'

    def _extract_title(self, file_name: str, content: str) -> str:
        """Extract title from filename or content"""
        # Clean filename as fallback title
        title = file_name.replace('_', ' ').replace('-', ' ')
        title = re.sub(r'\s+', ' ', title).strip()

        # Try to find title in content (first few lines)
        lines = content.split('\n')[:10]  # Check first 10 lines

        for line in lines:
            line = line.strip()
            # Look for title-like lines (not too long, not too short)
            if 10 <= len(line) <= 100 and not line.lower().startswith('page'):
                # Check if it looks like a title (capitalized, etc.)
                if line[0].isupper() and not line.endswith('.'):
                    title = line
                    break

        return title

    def _extract_sections(self, content: str, file_path: Path) -> List[Dict[str, str]]:
        """Extract sections from PDF content"""
        sections = []

        # Split by page breaks first
        pages = content.split('[PAGE BREAK]')

        for page_num, page_content in enumerate(pages, 1):
            if not page_content.strip():
                continue

            # Try to find section headers within each page
            page_sections = self._find_sections_in_text(page_content)

            if page_sections:
                sections.extend(page_sections)
            else:
                # If no clear sections, treat entire page as one section
                sections.append({
                    'title': f'Page {page_num}',
                    'content': page_content.strip(),
                    'page_number': page_num
                })

        return sections

    def _find_sections_in_text(self, text: str) -> List[Dict[str, str]]:
        """Find section headers in text"""
        sections = []
        lines = text.split('\n')

        current_section = {'title': 'Content', 'content': ''}

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line looks like a section header
            if self._is_section_header(line):
                # Save previous section
                if current_section['content'].strip():
                    sections.append(current_section.copy())

                # Start new section
                current_section = {
                    'title': line,
                    'content': ''
                }
            else:
                current_section['content'] += line + '\n'

        # Add final section
        if current_section['content'].strip():
            sections.append(current_section)

        return sections

    def _is_section_header(self, line: str) -> bool:
        """Determine if a line is likely a section header"""
        # Common section header patterns
        header_patterns = [
            r'^\d+\.\s+[A-Z]',  # "1. Introduction"
            r'^[IVX]+\.\s+[A-Z]',  # "I. Introduction"
            r'^[A-Z][A-Z\s]{2,20}$',  # "INTRODUCTION"
            r'^[A-Z][a-z]+ [A-Z][a-z]+',  # "Project Overview"
        ]

        # Check patterns
        for pattern in header_patterns:
            if re.match(pattern, line):
                return True

        # Additional heuristics
        if (len(line) < 50 and
            line[0].isupper() and
            not line.endswith('.') and
            any(keyword in line.lower() for keyword in
                ['introduction', 'conclusion', 'methodology', 'results', 'abstract',
                 'background', 'implementation', 'evaluation', 'discussion'])):
            return True

        return False

    def _generate_id(self, file_path: Path) -> str:
        """Generate unique ID for content"""
        try:
            # Try to get relative path from base_path
            relative_path = str(file_path.relative_to(self.base_path))
        except ValueError:
            # If file is not in base_path (like mingresume.pdf in raw folder),
            # just use the filename
            relative_path = file_path.name

        return relative_path.replace('/', '_').replace('.pdf', '')

    def save_processed_content(self, output_path: str):
        """Save processed content to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_content, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved processed PDF content to {output_file}")
        print(f"📊 Processed {len(self.processed_content)} PDF files")


# Example usage
if __name__ == "__main__":
    import os

    # Get the project root directory (go up from src/processor/ to project root)
    current_dir = Path(__file__).parent  # src/processor/
    project_root = current_dir.parent.parent  # Go up two levels to project root

    # Set paths relative to project root
    notion_export_path = project_root / "data" / "raw" / "notion_export"
    output_path = project_root / "data" / "processed" / "pdf_content.json"

    print(f"Looking for PDFs in: {notion_export_path}")
    print(f"Also checking: {notion_export_path.parent} (for files like mingresume.pdf)")
    print(f"Will save to: {output_path}")

    # Check if the notion export path exists
    if not notion_export_path.exists():
        print(f"❌ Notion export path does not exist: {notion_export_path}")
        print("Will only process PDFs in data/raw/ folder")
        # Use the raw folder instead
        raw_folder = project_root / "data" / "raw"
        processor = PDFProcessor(str(raw_folder))
    else:
        # Initialize processor with notion export path
        processor = PDFProcessor(str(notion_export_path))

    # Process all PDF files
    processed_content = processor.process_all_pdfs()

    # Save results
    processor.save_processed_content(str(output_path))

    # Print summary
    print("\n📋 PDF Processing Summary:")
    categories = {}
    total_pages = 0

    for content in processed_content:
        cat = content['metadata']['category']
        categories[cat] = categories.get(cat, 0) + 1
        total_pages += content.get('page_count', 0)

    for category, count in categories.items():
        print(f"  {category}: {count} files")

    print(f"📄 Total pages processed: {total_pages}")
