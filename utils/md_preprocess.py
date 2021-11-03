import os
import subprocess
import sys
from binascii import a2b_base64
from pathlib import Path

from nbconvert.preprocessors import Preprocessor
from traitlets import Unicode

AUDIO = '<IPython.lib.display.Audio object>'


class AudioPreprocessor(Preprocessor):
    output_filename_template = Unicode(
        "{unique_key}_{cell_index}_{index}{extension}"
    ).tag(config=True)

    def preprocess_cell(self, cell, resources, cell_index):
        unique_key = resources.get('unique_key', 'output')
        output_files_dir = resources.get('output_files_dir', None)

        for idx, out in enumerate(cell.get('outputs', [])):
            if out.output_type in {'execute_result', 'display_data'}:
                if not out.data['text/plain'] == AUDIO:
                    continue
                html_text = out.data['text/html']
                base64 = html_text.split('base64,')[1]
                base64, mime_type = base64.split('\" type=\"')
                mime_type = mime_type.split('\"')[0]
                data = a2b_base64(base64)
                ext = '.' + mime_type.split('/')[-1]

                filename = self.output_filename_template.format(
                    unique_key=unique_key,
                    cell_index=cell_index,
                    index=idx,
                    extension=ext)
                filename = os.path.join(output_files_dir, filename)

                out.data['text/html'] = f'![wav]({filename})'
                out.metadata.setdefault('filenames', {})
                out.metadata['filenames'][mime_type] = filename
                resources['outputs'][filename] = data

        return cell, resources


def remove_path(path: Path):
    if path.is_file() or path.is_symlink():
        path.unlink()
        return
    for p in path.iterdir():
        remove_path(p)
    path.rmdir()


# Publish notebook to my Obsidian Vault
def main(filepath: Path):
    destination = Path('~/Documents/Notes/Notes').expanduser()
    command = ['python', '-m', 'nbconvert', '--to', 'markdown',
               '--Exporter.preprocessors=["utils.md_preprocess.AudioPreprocessor"]',
               filepath]
    subprocess.run(command)

    attachments_dir = filepath.with_name(f'{filepath.stem}_files')
    md_file = filepath.with_suffix('.md')

    dest_att_dir = destination / 'attachments' / attachments_dir.name
    remove_path(dest_att_dir)
    attachments_dir.replace(dest_att_dir)
    md_file.replace(destination / md_file.name)


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) != 1:
        raise ValueError('Give only one notebook path.')
    main(Path(args[0]))
