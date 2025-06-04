# src/processors/markdown_processor.py

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any
import markdown
from markdown.extensions import codehilite, tables, toc

class MarkdownProcessor:
    """Process markdown files from Notion export"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.processed_content = []

    def process_all_markdown(self) -> List[Dict[str, Any]]:
        """Process all markdown files in the export"""
        print("🔍 Finding markdown files...")

        # Find all .md files recursively
        md_files = list(self.base_path.rglob("*.md"))
        print(f"Found {len(md_files)} markdown files")

        for md_file in md_files:
            try:
                content = self.process_single_file(md_file)
                if content:
                    self.processed_content.append(content)
                    print(f"✅ Processed: {md_file.name}")
            except Exception as e:
                print(f"❌ Error processing {md_file.name}: {e}")

        return self.processed_content

    def process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Process a single markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_content = f.read()

        # Extract metadata from file path and name
        metadata = self._extract_metadata(file_path, raw_content)

        # Clean and process content
        cleaned_content = self._clean_content(raw_content)

        # Convert markdown to structured text
        html_content = markdown.markdown(
            cleaned_content,
            extensions=['tables', 'toc', 'codehilite']
        )

        # Extract plain text (removing HTML tags)
        plain_text = self._html_to_text(html_content)

        # Extract sections
        sections = self._extract_sections(cleaned_content)

        return {
            'id': self._generate_id(file_path),
            'source_file': str(file_path),
            'type': 'markdown',
            'title': metadata['title'],
            'category': metadata['category'],
            'raw_content': raw_content,
            'cleaned_content': cleaned_content,
            'plain_text': plain_text,
            'sections': sections,
            'metadata': metadata,
            'word_count': len(plain_text.split()),
            'referenced_files': self._extract_file_references(raw_content)
        }

    def _extract_metadata(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Extract metadata from file path and content"""
        file_name = file_path.stem

        # Determine content type based on file path and content
        if 'Contact' in file_name:
            category = 'contact'
            title = 'Contact Information'
        elif file_path.parent.name == 'Projects aee4f34d22394d1690473e4917727405':
            category = 'project'
            title = self._clean_project_title(file_name)
        elif 'Portfolio' in file_name:
            category = 'portfolio_main'
            title = 'Portfolio Overview'
        else:
            category = 'general'
            title = self._clean_title(file_name)

        # Try to extract title from content (first header)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()

        return {
            'title': title,
            'category': category,
            'file_name': file_name,
            'relative_path': str(file_path.relative_to(self.base_path)),
            'last_modified': file_path.stat().st_mtime if file_path.exists() else None
        }

    def _clean_content(self, content: str) -> str:
        """Clean markdown content"""
        # Remove Notion-specific artifacts
        content = re.sub(r'%20', ' ', content)  # URL encoding
        content = re.sub(r'%E2%80%99', "'", content)  # Smart quotes

        # Clean up file references but keep meaningful ones
        content = re.sub(r'\[([^\]]+)\]\([^)]+\.(png|jpg|jpeg|gif|mp4|mov)\)', r'[Image: \1]', content)

        # Normalize whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)  # Multiple newlines
        content = re.sub(r'[ \t]+', ' ', content)  # Multiple spaces

        return content.strip()

    def _extract_sections(self, content: str) -> List[Dict[str, str]]:
        """Extract sections based on headers"""
        sections = []
        current_section = {'title': 'Introduction', 'content': ''}

        lines = content.split('\n')

        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                # Save previous section if it has content
                if current_section['content'].strip():
                    sections.append(current_section.copy())

                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {
                    'title': title,
                    'content': '',
                    'level': level
                }
            else:
                current_section['content'] += line + '\n'

        # Add the last section
        if current_section['content'].strip():
            sections.append(current_section)

        return sections

    def _html_to_text(self, html_content: str) -> str:
        """Convert HTML to plain text"""
        # Simple HTML tag removal
        text = re.sub(r'<[^>]+>', '', html_content)
        # Decode HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_file_references(self, content: str) -> List[str]:
        """Extract referenced files (PDFs, images, etc.)"""
        # Find file references in markdown links
        file_refs = re.findall(r'\[([^\]]+)\]\(([^)]+\.(pdf|png|jpg|jpeg|gif|mp4|mov))\)', content)
        return [{'name': ref[0], 'path': ref[1], 'type': ref[2]} for ref in file_refs]

    def _clean_project_title(self, file_name: str) -> str:
        """Clean project titles from Notion export format"""
        # Remove Notion ID hash
        title = re.sub(r'\s+[a-f0-9]{32}$', '', file_name)
        title = re.sub(r'[a-f0-9]{32}$', '', title)

        # Clean up common patterns
        title = title.replace('_', ' ')
        title = re.sub(r'\s+', ' ', title)

        return title.strip()

    def _clean_title(self, title: str) -> str:
        """General title cleaning"""
        title = title.replace('_', ' ')
        title = re.sub(r'\s+[a-f0-9]{32}$', '', title)  # Remove Notion IDs
        title = re.sub(r'\s+', ' ', title)
        return title.strip()

    def _generate_id(self, file_path: Path) -> str:
        """Generate unique ID for content"""
        relative_path = str(file_path.relative_to(self.base_path))
        return relative_path.replace('/', '_').replace('.md', '')

    def save_processed_content(self, output_path: str):
        """Save processed content to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_content, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved processed markdown content to {output_file}")
        print(f"📊 Processed {len(self.processed_content)} markdown files")


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = MarkdownProcessor("data/raw/notion_export")

    # Process all markdown files
    processed_content = processor.process_all_markdown()

    # Save results
    processor.save_processed_content("data/processed/markdown_content.json")

    # Print summary
    print("\n📋 Processing Summary:")
    categories = {}
    for content in processed_content:
        cat = content['metadata']['category']
        categories[cat] = categories.get(cat, 0) + 1

    for category, count in categories.items():
        print(f"  {category}: {count} files")
