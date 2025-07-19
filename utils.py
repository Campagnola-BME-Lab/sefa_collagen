"""Utility functions."""

import base64
import os
import subprocess
import cv2
import numpy as np

import torch

from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type

__all__ = ['postprocess', 'load_generator', 'factorize_weight',
           'HtmlPageVisualizer']

CHECKPOINT_DIR = 'checkpoints'


def to_tensor(array):
    """Converts a `numpy.ndarray` to `torch.Tensor`.

    Args:
      array: The input array to convert.

    Returns:
      A `torch.Tensor` with dtype `torch.FloatTensor` on cuda device.
    """
    assert isinstance(array, np.ndarray)
    return torch.from_numpy(array).type(torch.FloatTensor).cuda()


def postprocess(images, min_val=-1.0, max_val=1.0):
    """Post-processes images from `torch.Tensor` to `numpy.ndarray`.

    Args:
        images: A `torch.Tensor` with shape `NCHW` to process.
        min_val: The minimum value of the input tensor. (default: -1.0)
        max_val: The maximum value of the input tensor. (default: 1.0)

    Returns:
        A `numpy.ndarray` with shape `NHWC` and pixel range [0, 255].
    """
    assert isinstance(images, torch.Tensor)
    images = images.detach().cpu().numpy()
    images = (images - min_val) * 255 / (max_val - min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    images = images.transpose(0, 2, 3, 1)
    return images


def load_generator(model_name, device='cuda'):
    """Loads pre-trained generator.

    Args:
        model_name: Name of the model. Should be a key in `models.MODEL_ZOO`.

    Returns:
        A generator, which is a `torch.nn.Module`, with pre-trained weights
            loaded.

    Raises:
        KeyError: If the input `model_name` is not in `models.MODEL_ZOO`.
    """
    import dnnlib
    import legacy
    from torch_utils.persistence import persistent_class
    from torch_utils.persistence import _reconstruct_persistent_obj
    from models.stylegan2_ada import Generator  # Import your custom Generator class
    
    if model_name not in MODEL_ZOO:
        raise KeyError(f'Unknown model name `{model_name}`!')

    model_config = MODEL_ZOO[model_name].copy()
    url = model_config.pop('url', None)  # URL to download model if needed.
    
    # Determine checkpoint path
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name + '.pkl')
    
    # Download checkpoint if URL is provided
    if url:
        print(f'Loading checkpoint from `{url}` ...')
        if not os.path.exists(checkpoint_path):
            print(f'  Downloading checkpoint from `{url}` ...')
            subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
            print(f'  Finish downloading checkpoint.')
    else:
        print(f'No URL found for model `{model_name}`, checking local checkpoint.')

    # Build generator.
    print(f'Building generator for model `{model_name}` ...')
    generator = build_generator(**model_config)
    print(f'Finish building generator.')

    # Load pre-trained weights.
    # Ensure checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f'Checkpoint file `{checkpoint_path}` does not exist and no URL is provided!'
        )

    # Load generator from `.pkl` file
    print(f'Loading generator from `{checkpoint_path}` ...')
    #with open(checkpoint_path, 'rb') as f:
    #   checkpoint = legacy.load_network_pkl(f)  # Load StyleGAN2-ADA-PyTorch model

    #generator = checkpoint['G_ema']  # Use the EMA version of the generator
    generator = Generator(checkpoint_path, device=device)
    
    #Debugging statements
    #print(f"Generator type: {type(generator)}")
    #print(f"Generator attributes: {dir(generator)}")

        # Reconstruct persistent object if wrapped
    if hasattr(generator, '_persistent') or 'persistent_class' in str(type(generator)):
        print("Reconstructing persistent object...")
        generator = generator.__class__.__bases__[0](**generator.init_kwargs)
    
    generator.gan_type = 'stylegan2_ada'
    generator.__class__ = Generator

    # Add the `num_layers` attribute
    #generator.num_layers = len(list(generator.G.synthesis.children()))    
    #generator.latent_dim = generator.z_dim
    #generator.z_space_dim = generator.z_dim  # Alias for compatibility with Sefa
    #generator.c_dim = generator.c_dim
    
    generator = generator.cuda()
    generator.eval()  # Set to evaluation mode
    print(f'Finish loading generator from `{checkpoint_path}`.')
    return generator


def parse_indices(obj, min_val=None, max_val=None):
    """Parses indices.

    The input can be a list or a tuple or a string, which is either a comma
    separated list of numbers 'a, b, c', or a dash separated range 'a - c'.
    Space in the string will be ignored.

    Args:
        obj: The input object to parse indices from.
        min_val: If not `None`, this function will check that all indices are
            equal to or larger than this value. (default: None)
        max_val: If not `None`, this function will check that all indices are
            equal to or smaller than this value. (default: None)

    Returns:
        A list of integers.

    Raises:
        If the input is invalid, i.e., neither a list or tuple, nor a string.
    """
    if obj is None or obj == '':
        indices = []
    elif isinstance(obj, int):
        indices = [obj]
    elif isinstance(obj, (list, tuple, np.ndarray)):
        indices = list(obj)
    elif isinstance(obj, str):
        indices = []
        splits = obj.replace(' ', '').split(',')
        for split in splits:
            numbers = list(map(int, split.split('-')))
            if len(numbers) == 1:
                indices.append(numbers[0])
            elif len(numbers) == 2:
                indices.extend(list(range(numbers[0], numbers[1] + 1)))
            else:
                raise ValueError(f'Unable to parse the input!')

    else:
        raise ValueError(f'Invalid type of input: `{type(obj)}`!')

    assert isinstance(indices, list)
    indices = sorted(list(set(indices)))
    for idx in indices:
        assert isinstance(idx, int)
        if min_val is not None:
            assert idx >= min_val, f'{idx} is smaller than min val `{min_val}`!'
        if max_val is not None:
            assert idx <= max_val, f'{idx} is larger than max val `{max_val}`!'

    return indices


def factorize_weight(generator, layer_idx='all'):
    """Factorizes the generator weight to get semantics boundaries.

    Args:
        generator: Generator to factorize.
        layer_idx: Indices of layers to interpret, especially for StyleGAN and
            StyleGAN2. (default: `all`)

    Returns:
        A tuple of (layers_to_interpret, semantic_boundaries, eigen_values).

    Raises:
        ValueError: If the generator type is not supported.
    """
    # Get GAN type.
    gan_type = parse_gan_type(generator)
    #Debug statements
    #print(f"Generator type: {type(generator)}")
    #print(f"Generator attributes: {dir(generator)}")

    # Get layers.
    if gan_type == 'pggan':
        layers = [0]
    elif gan_type in ['stylegan', 'stylegan2', 'stylegan2_ada']:
        if layer_idx == 'all':
            layers = list(range(generator.num_layers))
        else:
            layers = parse_indices(layer_idx, min_val=0, max_val=generator.num_layers - 1)

    weights = []
    
    for idx in layers:
        layer = list(generator.G.synthesis.children())[idx]
        print(f"Attributes of layer {idx}: {dir(layer)}")

        if gan_type == 'stylegan2_ada':
            if hasattr(layer, 'conv1'):
                weight = layer.conv1.weight.T.cpu().detach().numpy()
                print(f"Using weight from layer {idx}: conv1, shape {weight.shape}")
            elif hasattr(layer, 'torgb'):
                weight = layer.torgb.weight.T.cpu().detach().numpy()
                print(f"Using weight from layer {idx}: torgb, shape {weight.shape}")
            else:
                raise AttributeError(f"Cannot find conv1 or torgb in layer {idx}.")
                
        elif gan_type == 'stylegan2' and idx == generator.num_layers - 1:
            weight = generator.synthesis.__getattr__(f'output{idx // 2}').style.weight.T.cpu().detach().numpy()
        elif gan_type == 'pggan':
            weight = generator.__getattr__(f'layer{idx}').weight.flip(2, 3).permute(1, 0, 2, 3).flatten(1).cpu().detach().numpy()
        elif gan_type == 'stylegan':
            weight = generator.synthesis.__getattr__(f'layer{idx}').style.weight.T.cpu().detach().numpy()
        
        weight = weight.reshape(weight.shape[0], -1)
        print(f"Processed weight from layer {idx}: shape {weight.shape}")
        weights.append(weight)

    # Ensure all weights have same feature dimension
    feature_dim = max(weight.shape[1] for weight in weights)
    
    latent_dim = generator.z_dim #usually 512
    print(f"Ensure latent dim is 512: {latent_dim}")
    
    aligned_weights = []
    for weight in weights:
        if weight.shape[1] < latent_dim:
            padding = np.zeros((weight.shape[0], latent_dim - weight.shape[1]))
            aligned_weight = np.concatenate([weight, padding], axis=1)
        elif weight.shape[1] > latent_dim:
            aligned_weight = weight[:, :latent_dim]
        else:
            aligned_weight = weight
        aligned_weights.append(aligned_weight)

    # Now stack them
    combined_weight = np.concatenate(aligned_weights, axis=0).astype(np.float32)
    print(f"Final combined weight shape: {combined_weight.shape}")

    # Normalize and factorize
    combined_weight /= np.linalg.norm(combined_weight, axis=0, keepdims=True)
    covariance_matrix = combined_weight.T @ combined_weight  # Shape (512, 512)
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)

    # Sort descending
    idx = np.argsort(-eigen_values)
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    boundaries = eigen_vectors.T  # Shape (512, 512)
    print(f"Final boundaries shape: {boundaries.shape}")

    return layers, eigen_vectors.T, eigen_values


def get_sortable_html_header(column_name_list, sort_by_ascending=False):
    """Gets header for sortable html page.

    Basically, the html page contains a sortable table, where user can sort the
    rows by a particular column by clicking the column head.

    Example:

    column_name_list = [name_1, name_2, name_3]
    header = get_sortable_html_header(column_name_list)
    footer = get_sortable_html_footer()
    sortable_table = ...
    html_page = header + sortable_table + footer

    Args:
        column_name_list: List of column header names.
        sort_by_ascending: Default sorting order. If set as `True`, the html
            page will be sorted by ascending order when the header is clicked
            for the first time.

    Returns:
        A string, which represents for the header for a sortable html page.
    """
    header = '\n'.join([
        '<script type="text/javascript">',
        'var column_idx;',
        'var sort_by_ascending = ' + str(sort_by_ascending).lower() + ';',
        '',
        'function sorting(tbody, column_idx){',
        '    this.column_idx = column_idx;',
        '    Array.from(tbody.rows)',
        '             .sort(compareCells)',
        '             .forEach(function(row) { tbody.appendChild(row); })',
        '    sort_by_ascending = !sort_by_ascending;',
        '}',
        '',
        'function compareCells(row_a, row_b) {',
        '    var val_a = row_a.cells[column_idx].innerText;',
        '    var val_b = row_b.cells[column_idx].innerText;',
        '    var flag = sort_by_ascending ? 1 : -1;',
        '    return flag * (val_a > val_b ? 1 : -1);',
        '}',
        '</script>',
        '',
        '<html>',
        '',
        '<head>',
        '<style>',
        '    table {',
        '        border-spacing: 0;',
        '        border: 1px solid black;',
        '    }',
        '    th {',
        '        cursor: pointer;',
        '    }',
        '    th, td {',
        '        text-align: left;',
        '        vertical-align: middle;',
        '        border-collapse: collapse;',
        '        border: 0.5px solid black;',
        '        padding: 8px;',
        '    }',
        '    tr:nth-child(even) {',
        '        background-color: #d2d2d2;',
        '    }',
        '</style>',
        '</head>',
        '',
        '<body>',
        '',
        '<table>',
        '<thead>',
        '<tr>',
        ''])
    for idx, name in enumerate(column_name_list):
        header += f'    <th onclick="sorting(tbody, {idx})">{name}</th>\n'
    header += '</tr>\n'
    header += '</thead>\n'
    header += '<tbody id="tbody">\n'

    return header


def get_sortable_html_footer():
    """Gets footer for sortable html page.

    Check function `get_sortable_html_header()` for more details.
    """
    return '</tbody>\n</table>\n\n</body>\n</html>\n'


def parse_image_size(obj):
    """Parses object to a pair of image size, i.e., (width, height).

    Args:
        obj: The input object to parse image size from.

    Returns:
        A two-element tuple, indicating image width and height respectively.

    Raises:
        If the input is invalid, i.e., neither a list or tuple, nor a string.
    """
    if obj is None or obj == '':
        width = height = 0
    elif isinstance(obj, int):
        width = height = obj
    elif isinstance(obj, (list, tuple, np.ndarray)):
        numbers = tuple(obj)
        if len(numbers) == 0:
            width = height = 0
        elif len(numbers) == 1:
            width = height = numbers[0]
        elif len(numbers) == 2:
            width = numbers[0]
            height = numbers[1]
        else:
            raise ValueError(f'At most two elements for image size.')
    elif isinstance(obj, str):
        splits = obj.replace(' ', '').split(',')
        numbers = tuple(map(int, splits))
        if len(numbers) == 0:
            width = height = 0
        elif len(numbers) == 1:
            width = height = numbers[0]
        elif len(numbers) == 2:
            width = numbers[0]
            height = numbers[1]
        else:
            raise ValueError(f'At most two elements for image size.')
    else:
        raise ValueError(f'Invalid type of input: {type(obj)}!')

    return (max(0, width), max(0, height))


def encode_image_to_html_str(image, image_size=None):
    """Encodes an image to html language.
    NOTE: Input image is always assumed to be with `RGB` channel order.
    Args:
        image: The input image to encode. Should be with `RGB` channel order.
        image_size: This field is used to resize the image before encoding. `0`
            disables resizing. (default: None)
    Returns:
        A string which represents the encoded image.
    """
    if image is None:
        return ''

    assert image.ndim == 3 and image.shape[2] in [1, 3]

    # Change channel order to `BGR`, which is opencv-friendly.
    image = image[:, :, ::-1]

    # Resize the image if needed.
    width, height = parse_image_size(image_size)
    if height or width:
        height = height or image.shape[0]
        width = width or image.shape[1]
        image = cv2.resize(image, (width, height))

    # Encode the image to html-format string.
    encoded_image = cv2.imencode('.jpg', image)[1].tostring()
    encoded_image_base64 = base64.b64encode(encoded_image).decode('utf-8')
    html_str = f'<img src="data:image/jpeg;base64, {encoded_image_base64}"/>'

    return html_str


def get_grid_shape(size, row=0, col=0, is_portrait=False):
    """Gets the shape of a grid based on the size.

    This function makes greatest effort on making the output grid square if
    neither `row` nor `col` is set. If `is_portrait` is set as `False`, the
    height will always be equal to or smaller than the width. For example, if
    input `size = 16`, output shape will be `(4, 4)`; if input `size = 15`,
    output shape will be (3, 5). Otherwise, the height will always be equal to
    or larger than the width.

    Args:
        size: Size (height * width) of the target grid.
        is_portrait: Whether to return a portrait size of a landscape size.
            (default: False)

    Returns:
        A two-element tuple, representing height and width respectively.
    """
    assert isinstance(size, int)
    assert isinstance(row, int)
    assert isinstance(col, int)
    if size == 0:
        return (0, 0)

    if row > 0 and col > 0 and row * col != size:
        row = 0
        col = 0

    if row > 0 and size % row == 0:
        return (row, size // row)
    if col > 0 and size % col == 0:
        return (size // col, col)

    row = int(np.sqrt(size))
    while row > 0:
        if size % row == 0:
            col = size // row
            break
        row = row - 1

    return (col, row) if is_portrait else (row, col)


class HtmlPageVisualizer(object):
    """Defines the html page visualizer.

    This class can be used to visualize image results as html page. Basically,
    it is based on an html-format sorted table with helper functions
    `get_sortable_html_header()`, `get_sortable_html_footer()`, and
    `encode_image_to_html_str()`. To simplify the usage, specifying the
    following fields are enough to create a visualization page:

    (1) num_rows: Number of rows of the table (header-row exclusive).
    (2) num_cols: Number of columns of the table.
    (3) header contents (optional): Title of each column.

    NOTE: `grid_size` can be used to assign `num_rows` and `num_cols`
    automatically.

    Example:

    html = HtmlPageVisualizer(num_rows, num_cols)
    html.set_headers([...])
    for i in range(num_rows):
        for j in range(num_cols):
            html.set_cell(i, j, text=..., image=..., highlight=False)
    html.save('visualize.html')
    """

    def __init__(self,
                 num_rows=0,
                 num_cols=0,
                 grid_size=0,
                 is_portrait=True,
                 viz_size=None):
        if grid_size > 0:
            num_rows, num_cols = get_grid_shape(
                grid_size, row=num_rows, col=num_cols, is_portrait=is_portrait)
        assert num_rows > 0 and num_cols > 0

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.viz_size = parse_image_size(viz_size)
        self.headers = ['' for _ in range(self.num_cols)]
        self.cells = [[{
            'text': '',
            'image': '',
            'highlight': False,
        } for _ in range(self.num_cols)] for _ in range(self.num_rows)]

    def set_header(self, col_idx, content):
        """Sets the content of a particular header by column index."""
        self.headers[col_idx] = content

    def set_headers(self, contents):
        """Sets the contents of all headers."""
        if isinstance(contents, str):
            contents = [contents]
        assert isinstance(contents, (list, tuple))
        assert len(contents) == self.num_cols
        for col_idx, content in enumerate(contents):
            self.set_header(col_idx, content)

    def set_cell(self, row_idx, col_idx, text='', image=None, highlight=False):
        """Sets the content of a particular cell.

        Basically, a cell contains some text as well as an image. Both text and
        image can be empty.

        Args:
            row_idx: Row index of the cell to edit.
            col_idx: Column index of the cell to edit.
            text: Text to add into the target cell. (default: None)
            image: Image to show in the target cell. Should be with `RGB`
                channel order. (default: None)
            highlight: Whether to highlight this cell. (default: False)
        """
        self.cells[row_idx][col_idx]['text'] = text
        self.cells[row_idx][col_idx]['image'] = encode_image_to_html_str(
            image, self.viz_size)
        self.cells[row_idx][col_idx]['highlight'] = bool(highlight)

    def save(self, save_path):
        """Saves the html page."""
        html = ''
        for i in range(self.num_rows):
            html += f'<tr>\n'
            for j in range(self.num_cols):
                text = self.cells[i][j]['text']
                image = self.cells[i][j]['image']
                if self.cells[i][j]['highlight']:
                    color = ' bgcolor="#FF8888"'
                else:
                    color = ''
                if text:
                    html += f'    <td{color}>{text}<br><br>{image}</td>\n'
                else:
                    html += f'    <td{color}>{image}</td>\n'
            html += f'</tr>\n'

        header = get_sortable_html_header(self.headers)
        footer = get_sortable_html_footer()

        with open(save_path, 'w') as f:
            f.write(header + html + footer)
