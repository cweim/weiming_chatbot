# src/processors/content_aggregator.py

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class ContentChunk:
    """Represents a chunk of content for RAG"""
    id: str
    content: str
    metadata: Dict[str, Any]
    source_type: str
    source_file: str
    word_count: int

class ContentAggregator:
    """Aggregate and chunk all processed content for RAG"""

    def __init__(self, processed_data_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.processed_data_dir = Path(processed_data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.all_chunks = []

    def aggregate_all_content(self) -> List[ContentChunk]:
        """Aggregate all processed content into chunks"""
        print("🔄 Aggregating all content...")

        # Load markdown content
        self._load_markdown_content()

        # Load PDF content
        self._load_pdf_content()

        # Load CSV content
        self._load_csv_content()

        # Load basic info
        self._load_basic_info()

        print(f"✅ Created {len(self.all_chunks)} content chunks")
        return self.all_chunks

    def _load_markdown_content(self):
        """Load and chunk markdown content"""
        md_file = self.processed_data_dir / "markdown_content.json"
        if not md_file.exists():
            print("⚠️  No markdown content found")
            return

        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = json.load(f)

        print(f"📝 Processing {len(md_content)} markdown files...")

        for item in md_content:
            # Create chunks for each markdown file
            if item['metadata']['category'] == 'portfolio_main':
                # Portfolio main page - chunk by sections
                self._chunk_portfolio_main(item)
            elif item['metadata']['category'] == 'project':
                # Project pages - chunk by project
                self._chunk_project_content(item)
            elif item['metadata']['category'] == 'contact':
                # Contact info - single chunk
                self._create_contact_chunk(item)
            else:
                # General content - chunk by text length
                self._chunk_general_content(item)

    def _load_pdf_content(self):
        """Load and chunk PDF content"""
        pdf_file = self.processed_data_dir / "pdf_content.json"
        if not pdf_file.exists():
            print("⚠️  No PDF content found")
            return

        with open(pdf_file, 'r', encoding='utf-8') as f:
            pdf_content = json.load(f)

        print(f"📄 Processing {len(pdf_content)} PDF files...")

        for item in pdf_content:
            if item['metadata']['category'] == 'project_report':
                self._chunk_project_report(item)
            elif item['metadata']['category'] == 'resume':
                self._chunk_resume(item)
            elif item['metadata']['category'] == 'presentation':
                self._chunk_presentation(item)
            else:
                self._chunk_general_pdf(item)

    def _load_csv_content(self):
        """Load and process CSV content"""
        # Look for CSV files in the raw data
        try:
            csv_files = list((self.processed_data_dir.parent / "raw" / "notion_export").rglob("*.csv"))

            if csv_files:
                print(f"📊 Found {len(csv_files)} CSV files")
                for csv_file in csv_files:
                    try:
                        if 'Projects' in csv_file.name:
                            self._process_projects_csv(csv_file)
                    except Exception as e:
                        print(f"❌ Error processing CSV {csv_file.name}: {e}")
            else:
                print("ℹ️  No CSV files found")
        except Exception as e:
            print(f"⚠️  Error searching for CSV files: {e}")

    def _load_basic_info(self):
        """Load basic info if available"""
        basic_info_file = self.processed_data_dir.parent / "raw" / "basic_info.json"
        if basic_info_file.exists():
            try:
                with open(basic_info_file, 'r', encoding='utf-8') as f:
                    basic_info = json.load(f)
                self._chunk_basic_info(basic_info)
                print(f"✅ Loaded basic info with {len(basic_info)} sections")
            except json.JSONDecodeError as e:
                print(f"⚠️  Error reading basic_info.json: {e}")
                print(f"⚠️  Skipping basic info processing. Please check line {e.lineno} column {e.colno}")
                print("💡 Tip: Look for unescaped quotes, newlines, or special characters")
            except Exception as e:
                print(f"⚠️  Error processing basic_info.json: {e}")
        else:
            print("ℹ️  No basic_info.json found, skipping personal info processing")

    def _chunk_portfolio_main(self, item: Dict[str, Any]):
        """Chunk main portfolio page"""
        # Bio section
        bio_content = self._extract_bio_section(item['cleaned_content'])
        if bio_content:
            chunk = ContentChunk(
                id=f"{item['id']}_bio",
                content=bio_content,
                metadata={
                    'type': 'bio',
                    'category': 'personal_info',
                    'title': 'Professional Bio',
                    'source_title': item['title'],
                    'priority': 'high'
                },
                source_type='markdown',
                source_file=item['source_file'],
                word_count=len(bio_content.split())
            )
            self.all_chunks.append(chunk)

        # Certifications section
        cert_content = self._extract_certifications_section(item)
        if cert_content:
            chunk = ContentChunk(
                id=f"{item['id']}_certifications",
                content=cert_content,
                metadata={
                    'type': 'certifications',
                    'category': 'credentials',
                    'title': 'Certifications and Credentials',
                    'source_title': item['title'],
                    'priority': 'medium'
                },
                source_type='markdown',
                source_file=item['source_file'],
                word_count=len(cert_content.split())
            )
            self.all_chunks.append(chunk)

    def _chunk_project_content(self, item: Dict[str, Any]):
        """Chunk individual project content"""
        project_title = item['metadata']['title']
        content = item['cleaned_content']

        # For projects, create chunks by sections but keep project context
        if item['sections']:
            for i, section in enumerate(item['sections']):
                section_content = f"Project: {project_title}\n\n"
                section_content += f"Section: {section['title']}\n\n"
                section_content += section['content']

                chunk = ContentChunk(
                    id=f"{item['id']}_section_{i}",
                    content=section_content,
                    metadata={
                        'type': 'project_section',
                        'category': 'project',
                        'title': f"{project_title} - {section['title']}",
                        'project_name': project_title,
                        'section_name': section['title'],
                        'source_title': item['title'],
                        'priority': 'high'
                    },
                    source_type='markdown',
                    source_file=item['source_file'],
                    word_count=len(section_content.split())
                )
                self.all_chunks.append(chunk)
        else:
            # No sections, chunk the entire project content
            full_content = f"Project: {project_title}\n\n{content}"
            chunks = self._split_text_into_chunks(full_content)

            for i, chunk_text in enumerate(chunks):
                chunk = ContentChunk(
                    id=f"{item['id']}_chunk_{i}",
                    content=chunk_text,
                    metadata={
                        'type': 'project_content',
                        'category': 'project',
                        'title': project_title,
                        'project_name': project_title,
                        'chunk_index': i,
                        'source_title': item['title'],
                        'priority': 'high'
                    },
                    source_type='markdown',
                    source_file=item['source_file'],
                    word_count=len(chunk_text.split())
                )
                self.all_chunks.append(chunk)

    def _create_contact_chunk(self, item: Dict[str, Any]):
        """Create contact information chunk"""
        chunk = ContentChunk(
            id=f"{item['id']}_contact",
            content=f"Contact Information:\n\n{item['cleaned_content']}",
            metadata={
                'type': 'contact',
                'category': 'personal_info',
                'title': 'Contact Information',
                'source_title': item['title'],
                'priority': 'medium'
            },
            source_type='markdown',
            source_file=item['source_file'],
            word_count=len(item['cleaned_content'].split())
        )
        self.all_chunks.append(chunk)

    def _chunk_general_content(self, item: Dict[str, Any]):
        """Chunk general markdown content"""
        chunks = self._split_text_into_chunks(item['cleaned_content'])

        for i, chunk_text in enumerate(chunks):
            chunk = ContentChunk(
                id=f"{item['id']}_chunk_{i}",
                content=chunk_text,
                metadata={
                    'type': 'general',
                    'category': item['metadata']['category'],
                    'title': f"{item['title']} (Part {i+1})",
                    'chunk_index': i,
                    'source_title': item['title'],
                    'priority': 'medium'
                },
                source_type='markdown',
                source_file=item['source_file'],
                word_count=len(chunk_text.split())
            )
            self.all_chunks.append(chunk)

    def _chunk_project_report(self, item: Dict[str, Any]):
        """Chunk project report PDFs"""
        project_name = item['metadata'].get('project_name', 'Unknown Project')

        # Process sections if available
        if item['sections']:
            for i, section in enumerate(item['sections']):
                section_content = f"Project Report: {project_name}\n"
                section_content += f"Document: {item['metadata']['title']}\n\n"
                section_content += f"{section['title']}\n\n{section['content']}"

                chunk = ContentChunk(
                    id=f"{item['id']}_section_{i}",
                    content=section_content,
                    metadata={
                        'type': 'project_report_section',
                        'category': 'project_documentation',
                        'title': f"{project_name} Report - {section['title']}",
                        'project_name': project_name,
                        'document_type': 'report',
                        'section_name': section['title'],
                        'source_title': item['title'],
                        'priority': 'high'
                    },
                    source_type='pdf',
                    source_file=item['source_file'],
                    word_count=len(section_content.split())
                )
                self.all_chunks.append(chunk)
        else:
            # Chunk by size
            chunks = self._split_text_into_chunks(item['cleaned_content'])
            for i, chunk_text in enumerate(chunks):
                full_content = f"Project Report: {project_name}\n"
                full_content += f"Document: {item['metadata']['title']}\n\n"
                full_content += chunk_text

                chunk = ContentChunk(
                    id=f"{item['id']}_chunk_{i}",
                    content=full_content,
                    metadata={
                        'type': 'project_report',
                        'category': 'project_documentation',
                        'title': f"{project_name} Report (Part {i+1})",
                        'project_name': project_name,
                        'document_type': 'report',
                        'chunk_index': i,
                        'source_title': item['title'],
                        'priority': 'high'
                    },
                    source_type='pdf',
                    source_file=item['source_file'],
                    word_count=len(full_content.split())
                )
                self.all_chunks.append(chunk)

    def _chunk_resume(self, item: Dict[str, Any]):
        """Chunk resume content"""
        chunks = self._split_text_into_chunks(item['cleaned_content'])

        for i, chunk_text in enumerate(chunks):
            content = f"Resume/CV Content:\n\n{chunk_text}"

            chunk = ContentChunk(
                id=f"{item['id']}_chunk_{i}",
                content=content,
                metadata={
                    'type': 'resume',
                    'category': 'professional_background',
                    'title': f"Resume (Part {i+1})",
                    'document_type': 'resume',
                    'chunk_index': i,
                    'source_title': item['title'],
                    'priority': 'high'
                },
                source_type='pdf',
                source_file=item['source_file'],
                word_count=len(content.split())
            )
            self.all_chunks.append(chunk)

    def _chunk_presentation(self, item: Dict[str, Any]):
        """Chunk presentation content"""
        project_name = item['metadata'].get('project_name', 'Unknown Project')
        chunks = self._split_text_into_chunks(item['cleaned_content'])

        for i, chunk_text in enumerate(chunks):
            content = f"Presentation: {project_name}\n"
            content += f"Document: {item['metadata']['title']}\n\n"
            content += chunk_text

            chunk = ContentChunk(
                id=f"{item['id']}_chunk_{i}",
                content=content,
                metadata={
                    'type': 'presentation',
                    'category': 'project_documentation',
                    'title': f"{project_name} Presentation (Part {i+1})",
                    'project_name': project_name,
                    'document_type': 'presentation',
                    'chunk_index': i,
                    'source_title': item['title'],
                    'priority': 'medium'
                },
                source_type='pdf',
                source_file=item['source_file'],
                word_count=len(content.split())
            )
            self.all_chunks.append(chunk)

    def _chunk_general_pdf(self, item: Dict[str, Any]):
        """Chunk general PDF content"""
        chunks = self._split_text_into_chunks(item['cleaned_content'])

        for i, chunk_text in enumerate(chunks):
            chunk = ContentChunk(
                id=f"{item['id']}_chunk_{i}",
                content=chunk_text,
                metadata={
                    'type': 'document',
                    'category': item['metadata']['category'],
                    'title': f"{item['title']} (Part {i+1})",
                    'document_type': item['metadata']['category'],
                    'chunk_index': i,
                    'source_title': item['title'],
                    'priority': 'medium'
                },
                source_type='pdf',
                source_file=item['source_file'],
                word_count=len(chunk_text.split())
            )
            self.all_chunks.append(chunk)

    def _process_projects_csv(self, csv_file: Path):
        """Process projects CSV file"""
        try:
            df = pd.read_csv(csv_file)

            # Create a summary chunk from CSV data
            projects_summary = "Projects Overview:\n\n"

            for _, row in df.iterrows():
                project_info = ""
                for col, value in row.items():
                    if pd.notna(value) and str(value).strip():
                        project_info += f"{col}: {value}\n"
                projects_summary += project_info + "\n"

            chunk = ContentChunk(
                id=f"projects_csv_{csv_file.stem}",
                content=projects_summary,
                metadata={
                    'type': 'projects_overview',
                    'category': 'project_metadata',
                    'title': 'Projects Overview (CSV)',
                    'source_title': csv_file.name,
                    'priority': 'medium'
                },
                source_type='csv',
                source_file=str(csv_file),
                word_count=len(projects_summary.split())
            )
            self.all_chunks.append(chunk)

        except Exception as e:
            print(f"Error processing CSV {csv_file.name}: {e}")

    def _chunk_basic_info(self, basic_info: Dict[str, Any]):
        """Chunk basic info content"""
        for key, info in basic_info.items():
            if isinstance(info, dict) and 'your_answer' in info and info['your_answer']:
                content = f"{info['prompt']}\n\nAnswer: {info['your_answer']}"

                chunk = ContentChunk(
                    id=f"basic_info_{key}",
                    content=content,
                    metadata={
                        'type': 'basic_info',
                        'category': 'personal_info',
                        'title': key.replace('_', ' ').title(),
                        'info_type': key,
                        'source_title': 'Basic Information',
                        'priority': 'high'
                    },
                    source_type='manual',
                    source_file='basic_info.json',
                    word_count=len(content.split())
                )
                self.all_chunks.append(chunk)

    def _extract_bio_section(self, content: str) -> str:
        """Extract bio section from portfolio content"""
        lines = content.split('\n')
        bio_content = ""

        # Look for bio section specifically
        for i, line in enumerate(lines):
            if 'bio' in line.lower():
                # Found bio section, get content until next section
                bio_content = line + "\n"
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith('#') and j > i + 1:  # Next section
                        break
                    bio_content += lines[j] + "\n"
                break
            elif 'student' in line.lower() and 'engineer' in line.lower():
                # This looks like a bio line
                bio_content = line + "\n"
                # Get a few more lines for context
                for j in range(i + 1, min(i + 5, len(lines))):
                    if lines[j].strip() and not lines[j].startswith('#'):
                        bio_content += lines[j] + "\n"
                break

        return bio_content.strip()

    def _extract_certifications_section(self, item: Dict[str, Any]) -> str:
        """Extract certifications from portfolio"""
        content = item['cleaned_content']
        referenced_files = item.get('referenced_files', [])

        cert_content = "Professional Certifications:\n\n"

        # Look for certification section in text
        if 'certification' in content.lower():
            lines = content.split('\n')
            in_cert_section = False

            for line in lines:
                if 'certification' in line.lower():
                    in_cert_section = True
                    cert_content += line + "\n"
                elif in_cert_section and line.startswith('#'):
                    break
                elif in_cert_section:
                    cert_content += line + "\n"

        # Add info about certification images
        cert_images = [f for f in referenced_files if 'screenshot' in f.get('path', '').lower()]
        if cert_images:
            cert_content += f"\nCertification documents include {len(cert_images)} credential images.\n"
            # Add image names for context
            for img in cert_images[:5]:  # Limit to first 5
                cert_content += f"- {img.get('name', 'Certification image')}\n"

        return cert_content if len(cert_content) > 50 else ""

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        words = text.split()
        chunks = []

        if len(words) <= self.chunk_size:
            # Text is short enough, return as single chunk
            return [text]

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)

            # Move start position with overlap
            start = end - self.chunk_overlap
            if start >= len(words):
                break

        return chunks

    def save_chunks(self, output_path: str):
        """Save all chunks to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert chunks to serializable format
        chunks_data = []
        for chunk in self.all_chunks:
            chunks_data.append({
                'id': chunk.id,
                'content': chunk.content,
                'metadata': chunk.metadata,
                'source_type': chunk.source_type,
                'source_file': chunk.source_file,
                'word_count': chunk.word_count
            })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved {len(chunks_data)} chunks to {output_file}")

        # Print summary statistics
        self._print_chunk_summary()

    def _print_chunk_summary(self):
        """Print summary of created chunks"""
        print("\n📊 Chunk Summary:")

        # By type
        type_counts = {}
        category_counts = {}
        total_words = 0

        for chunk in self.all_chunks:
            chunk_type = chunk.metadata.get('type', 'unknown')
            category = chunk.metadata.get('category', 'unknown')

            type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
            category_counts[category] = category_counts.get(category, 0) + 1
            total_words += chunk.word_count

        print("\nBy Type:")
        for chunk_type, count in sorted(type_counts.items()):
            print(f"  {chunk_type}: {count}")

        print("\nBy Category:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}")

        print(f"\n📝 Total words in chunks: {total_words:,}")
        if len(self.all_chunks) > 0:
            print(f"📦 Average words per chunk: {total_words // len(self.all_chunks)}")


# Example usage
if __name__ == "__main__":
    # Initialize aggregator
    aggregator = ContentAggregator(
        processed_data_dir="data/processed",
        chunk_size=1000,  # Adjust based on your needs
        chunk_overlap=200
    )

    # Aggregate all content
    chunks = aggregator.aggregate_all_content()

    # Save chunks
    aggregator.save_chunks("data/processed/final_chunks.json")
