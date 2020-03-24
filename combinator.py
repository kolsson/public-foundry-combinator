#!/Users/krister/anaconda3/envs/public-foundry-combinator/bin/python

import os, tempfile, datetime
from pathlib import Path
import copy
import re
import base64

import logging
import warnings

import absl.app
from absl import flags
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
from flask_socketio import SocketIO
from bs4 import BeautifulSoup as soup
from svgpathtools import parse_path

import numpy as np
import PIL.PngImagePlugin
import PIL.ImageOps  
import PIL.ImageEnhance
from PIL import ImageDraw
import tensorflow as tf

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.utils import registry
from tensor2tensor.utils import contrib
from tensor2tensor.utils import trainer_lib
from tensor2tensor.layers import common_layers

from magenta.models import svg_vae

import fontforge
import svg_utils
from svg_t2t import generate_t2t_example


##########################################################################################
# STDOUT
##########################################################################################

# we could use colorama instead...

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

##########################################################################################
# PATHS
##########################################################################################
    
inferencepath = Path('./inference')

basepath = Path('.')
t2tpath = basepath/'t2t'

##########################################################################################
# LOGGING
##########################################################################################

# Suppress Flask's info logging.
log = logging.getLogger("werkzeug")
log.setLevel(logging.WARNING)

# shut up and play the hits
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# warnings.simplefilter("ignore")

##########################################################################################
# FLASK
##########################################################################################

# Reference: https://github.com/tensorflow/minigo/blob/master/minigui/serve.py

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, logger=log, engineio_logger=log)

##########################################################################################
# TF
##########################################################################################

# only needed if we use initialize_model_with_t2t
# tfe = tf.contrib.eager

Modes = tf.estimator.ModeKeys

tf.compat.v1.enable_eager_execution()

problem_name = 'glyph_azzn_problem'

##########################################################################################
# HERE WE GO
##########################################################################################

# for now we just use our initial model
# once we add model switching we may want to put this in a dictionary

with tf.io.gfile.GFile(os.fspath(t2tpath/'mean.npz'), 'rb') as f: mean_npz = np.load(f)
with tf.io.gfile.GFile(os.fspath(t2tpath/'stdev.npz'), 'rb') as f: stdev_npz = np.load(f)

# our old order
# glyphs = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"

# order used internally
glyphs = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# probably useless -- we need a more intelligent way of producing filled glyphs
glyphMaxVoids = {
    '0': 1,
    '4': 1,
    '6': 1,
    '8': 2,
    '9': 1,
    'A': 1,
    'B': 1,
    'D': 1,
    'O': 1,
    'P': 1,
    'Q': 1,
    'R': 1,
    'a': 1,
    'b': 1,
    'd': 1,
    'e': 1,
    'g': 1,
    'o': 1,
    'p': 1,
    'q': 1,
}

maxpaths = 50

# borrowed from glyphtracer

font_ascent = 1000
font_height = 780

##########################################################################################
# MODEL
##########################################################################################

# UNUSED -- HERE FOR REFERENCE

# def initialize_model_with_t2t(hparam_set, add_hparams, model_name, ckpt_dir):
#     """Returns an initialized model and hparams using our trained t2t data."""
#     
#     data_dir = os.fspath(t2tpath)
#     
#     tf.reset_default_graph()
# 
#     # create hparams and get glyphazzn problem definition
#     # we don't need to add data_dir=data_dir
#     hparams = trainer_lib.create_hparams(hparam_set, add_hparams, problem_name=problem_name)
#     problem = registry.problem(problem_name)
# 
#     # get model definition
#     ModelClass = registry.model(model_name)
#     model = ModelClass(hparams, mode=Modes.PREDICT, problem_hparams=hparams.problem_hparams)
# 
#     # create dataset iterator from problem definition
#     dataset = problem.dataset(Modes.PREDICT, dataset_split=Modes.TRAIN,
#         data_dir=data_dir, shuffle_files=False, hparams=hparams).batch(1)
#     iterator = tfe.Iterator(dataset)
# 
#     # finalize/initialize model
#     output, extra_losses = model(iterator.next())  # creates ops to be initialized
#     model.initialize_from_ckpt(ckpt_dir)  # initializes ops
# 
#     return model, hparams

def initialize_model_with_example(example, hparam_set, add_hparams, model_name, ckpt_dir):
    """Returns an initialized model and hparams using our trained t2t data."""
    
    tf.reset_default_graph()

    # create hparams and get glyphazzn problem definition
    # we don't need to add data_dir=data_dir
    hparams = trainer_lib.create_hparams(hparam_set, add_hparams, problem_name=problem_name)
    problem = registry.problem(problem_name)

    # get model definition
    ModelClass = registry.model(model_name)
    model = ModelClass(hparams, mode=Modes.PREDICT, problem_hparams=hparams.problem_hparams)

    # finalize/initialize model
    features1 = preprocess_example(example, hparams) # passed by reference
    
    output, extra_losses = model(features1)  # creates ops to be initialized
    model.initialize_from_ckpt(ckpt_dir)  # initializes ops

    return model, hparams, features1

def get_bottleneck(features, model):
    """Retrieve latent encoding for given input pixel image in features dict."""
    features = features.copy()

    # the presence of a 'bottleneck' feature with 0 dimensions indicates that the
    # model should return the bottleneck from the input image
    features['bottleneck'] = tf.zeros((0, 128))

    return model(features)[0]

def infer_from_bottleneck(features, bottleneck, model, out='svg'):
    """Returns a sample from a decoder, conditioned on the given a latent."""
    features = features.copy()

    # set bottleneck which we're decoding from
    features['bottleneck'] = bottleneck

    # reset inputs/targets. This guarantees that the decoder is only being
    # conditioned on the given bottleneck.

    batch_size = tf.shape(bottleneck)[:1].numpy().tolist()
    features['inputs'] = tf.zeros(
        batch_size + tf.shape(features['inputs'])[1:].numpy().tolist())
    features['targets'] = tf.zeros(
        batch_size + tf.shape(features['targets'])[1:].numpy().tolist())
    features['targets_psr'] = tf.zeros(
        batch_size + tf.shape(features['targets_psr'])[1:].numpy().tolist())

    if out == 'svg':
        return model.infer(features, decode_length=0)
    else:
        return model(features)

##########################################################################################
# INFERENCE
##########################################################################################

def _tile(features, key, dims):
    """Helper that creates copies of features['keys'] across given dims."""

    features[key] = tf.tile(features[key], dims)
    return features

def decode_example(serialized_example):
    """Return a dict of Tensors from a serialized tensorflow.Example."""

    data_fields = {'targets_rel': tf.FixedLenFeature([51*10], tf.float32),
                   'targets_rnd': tf.FixedLenFeature([64*64], tf.float32),
                   'targets_sln': tf.FixedLenFeature([1], tf.int64),
                   'targets_cls': tf.FixedLenFeature([1], tf.int64)}

    # Necessary to rejoin examples in the correct order with the Cloud ML Engine
    # batch prediction API.
    
    data_fields["batch_prediction_key"] = tf.FixedLenFeature([1], tf.int64, 0)

    data_items_to_decoders = {
        field: contrib.slim().tfexample_decoder.Tensor(field) for field in data_fields
    }

    decoder = contrib.slim().tfexample_decoder.TFExampleDecoder(data_fields, data_items_to_decoders)

    decode_items = list(sorted(data_items_to_decoders))
    
    decoded = decoder.decode(serialized_example, items=decode_items)
    return dict(zip(decode_items, decoded))

def preprocess_example(example, hparams):
    """ Preprocess our example based on our magenta problem """

    example['targets_cls'] = tf.reshape(example['targets_cls'], [1])
    example['targets_sln'] = tf.reshape(example['targets_sln'], [1])

    example['targets_rel'] = tf.reshape(example['targets_rel'], [51, 1, 10])
    # normalize (via gaussian)
    example['targets_rel'] = (example['targets_rel'] - mean_npz) / stdev_npz

    # redefine shape inside model!
    example['targets_psr'] = tf.reshape(example['targets_rnd'], [1, 64 * 64]) / 255.
    del example['targets_rnd']

    if hparams.just_render:
        # training vae mode, use the last image (rendered icon) as input & output
        example['inputs'] = example['targets_psr'][-1, :]
        example['targets'] = example['targets_psr'][-1, :]
    else:
        example['inputs'] = tf.identity(example['targets_rel'])
        example['targets'] = tf.identity(example['targets_rel'])

    # our shaping

    example["batch_prediction_key"] = tf.expand_dims(example["batch_prediction_key"], 0)
    example["inputs"] = tf.expand_dims(example["inputs"], 0)
    example["targets"] = tf.expand_dims(example["targets"], 0)
    example["targets_cls"] = tf.expand_dims(example["targets_cls"], 0)
    example["targets_psr"] = tf.expand_dims(example["targets_psr"], 0)
    example["targets_rel"] = tf.expand_dims(example["targets_rel"], 0)
    example["targets_sln"] = tf.expand_dims(example["targets_sln"], 0)

    # we pass by reference so example is modified even if we don't return
    return example    

def infer_from_file(example_file, hparam_set, add_hparams, model_name, ckpt_dir, out='svg', bitmap_depth=8, bitmap_fill=False):
    """ Load, decode and infer our example """
    
    # https://www.tensorflow.org/tutorials/load_data/tfrecord
    raw_dataset = tf.data.TFRecordDataset([ os.fspath(example_file) ])

    for raw_record in raw_dataset.take(1):
        example = raw_record # we decode_example in infer()

    return infer(example, hparam_set, add_hparams, model_name, ckpt_dir, out, bitmap_depth, bitmap_fill)

def infer(example, hparam_set, add_hparams, model_name, ckpt_dir, out='svg', bitmap_depth=8, bitmap_fill=False):
    """Decodes one example of each class, conditioned on the example."""

    # initialize with t2t data
    # model, hparams = initialize_model_with_t2t(hparam_set, add_hparams, model_name, ckpt_dir)
    # features1 = preprocess_example(example, hparams) # passed by reference

    # OR initialize with example 
    model, hparams, features1 = initialize_model_with_example(decode_example(example), 
        hparam_set, add_hparams, model_name, ckpt_dir)

    # == the number of glyphs
    num_classes = hparams.num_categories 

    # get bottleneck of the features we selected before
    bottleneck1 = get_bottleneck(features1, model)
    bottleneck1 = tf.tile(bottleneck1, [num_classes, 1])

    # create class batch
    new_features = copy.copy(features1)

    clss_batch = tf.reshape([tf.constant([[clss]], dtype=tf.int64) 
        for clss in range(num_classes)], [-1, 1])
    new_features['targets_cls'] = clss_batch

    new_features = _tile(new_features, 'targets_psr', [num_classes, 1, 1])

    inp_target_dim = [num_classes, 1, 1, 1] if out == 'svg' else [num_classes, 1]

    new_features = _tile(new_features, 'inputs', inp_target_dim)
    new_features = _tile(new_features, 'targets', inp_target_dim)

    # run model
    output_batch = infer_from_bottleneck(new_features, bottleneck1, model, out)

    # render outputs to svg
    # (our inference example is features1['inputs'])
    output_batch = output_batch['outputs'] if out == 'svg' else output_batch[0]

    out_list = []    
    for i, output in enumerate(tf.split(output_batch, num_classes)):
        if out == 'svg':
            out_list.append(svg_render(output))
        elif out == 'img':
            out_list.append(bitmap_render(output, glyph=glyphs[i], depth=bitmap_depth, fill=bitmap_fill))
        else:
            out_list.append(bitmap_render(output, glyph=glyphs[i], depth=bitmap_depth, fill=bitmap_fill, render_html=False))
            
    return out_list

##########################################################################################
# Bitmap
##########################################################################################

def bitmap_render(tensor, glyph='0', depth=8, fill=False, render_html=True):
    """Converts Image VAE output into HTML svg."""
    
    # depth is not 1 we don't fill
    
    if not depth == 1:
        fill=False 
    
    # adapted from matplotlib code below:
    # imsave(tempbitmappath, np.reshape(tensor, [64, 64]), vmin=0, vmax=1, cmap='gray_r')

    arr = np.reshape(tensor, [64, 64])
    arr = np.clip(arr, 0, 1)
        
    arr = (arr * 255).round().astype(np.uint8)
 
    image = PIL.Image.fromarray(arr, "L")
    image = PIL.ImageOps.invert(image)
    enhancer = PIL.ImageEnhance.Contrast(image)
    image = enhancer.enhance(5)

    # other options for enhancement
    
    # image = PIL.ImageOps.autocontrast(image, cutoff=0, ignore=255)

    # enhancer = PIL.ImageEnhance.Sharpness(image)
    # image = enhancer.enhance(20)

    if depth == 1:
        # we can set this to darken (max should be < 0.5)
        darken = 0
    
        arr = (np.array(image).astype(np.float64)) / 255.0
        arr = arr - darken
        arr = np.around(arr)
        arr = (arr * 255).round().astype(np.uint8)
        image = PIL.Image.fromarray(arr, "L")

    # this is a stopgap -- we shouldn't be having to fill in our glyphs
    
#     if fill:
#         mask = image.copy()
#         image = image.convert("RGB")
#         
#         # fill black
#         ImageDraw.floodfill(mask, xy=(0, 0), value=0)
#     
#         if glyph in glyphMaxVoids:
#             pass
#         else:
#             # we don't expect any voids, just add our mask
#             image.paste((0, 0, 0), (0, 0), mask)
    
    if render_html:
        # create a temporary file

        tempbitmapfile = tempfile.NamedTemporaryFile(mode='w', suffix='.png', delete=False)
        tempbitmapfile.close()
        tempbitmappath = Path(tempbitmapfile.name)

        image.save(tempbitmappath, format="png")
        
        # load back and convert to html
    
        data_uri = base64.b64encode(tempbitmappath.read_bytes()).decode('utf-8')
        html = '<img src="data:image/png;base64,{0}">'.format(data_uri)
    
        # remove our temporary file
    
        tempbitmappath.unlink()

        return html
    else:
        return image
        
##########################################################################################
# SVG
##########################################################################################
  
def clean_and_center_svg(svg, glyph='', ymin=None, ymax=None, tag='path', flipv=False):
    # extract the first svg glyph path and bbox
        
    svg_tree = soup(svg, 'lxml')
    svg_tag = svg_tree.find(tag)
    
    path_obj = parse_path(svg_tag['d'])
    xmin, xmax, nymin, nymax = path_obj.bbox()
    
    ymin = min(ymin, nymin) if ymin is not None else nymin
    ymax = max(ymax, nymax) if ymax is not None else nymax
    
    print(f'{glyph}: {xmin}, {xmax}, {ymin}, {ymax}')
    
    svg_start_inputs = f'<svg width="50px" height="50px" viewBox="{xmin} {ymin} {xmax-xmin} {ymax-ymin}" version="1.1" xmlns="http://www.w3.org/2000/svg">'
    
    if flipv:
        svg_path = f'<path transform="scale(1, -1) translate(0, -{ymax+ymin})" d="{path_obj.d()}" />'
    else:
        svg_path = f'<path d="{path_obj.d()}" />'    
    
    return f'{svg_start_inputs}{svg_path}</svg>'

# Alternative: 

# Precompute ymin / ymax

def get_svg_path_ymin_ymax(svg, ymin=None, ymax=None, tag='path'):
    # extract the first svg glyph path and bbox
        
    svg_tree = soup(svg, 'lxml')
    svg_tag = svg_tree.find(tag)
    
    path_obj = parse_path(svg_tag['d'])
    xmin, xmax, nymin, nymax = path_obj.bbox()
    
    ymin = min(ymin, nymin) if ymin is not None else nymin
    ymax = max(ymax, nymax) if ymax is not None else nymax
    
    return (path_obj, ymin, ymax)

# Center based on min ymin / max ymax

def compose_svg(path_obj, ymin=None, ymax=None, flipv=False):
    xmin, xmax, _, _ = path_obj.bbox()

    svg_start_inputs = f'<svg width="50px" height="50px" viewBox="{xmin} {ymin} {xmax-xmin} {ymax-ymin}" version="1.1" xmlns="http://www.w3.org/2000/svg">'
    
    if flipv:
        svg_path = f'<path transform="scale(1, -1) translate(0, -{ymax+ymin})" d="{path_obj.d()}" />'
    else:
        svg_path = f'<path d="{path_obj.d()}" />'    
    
    return f'{svg_start_inputs}{svg_path}</svg>'   

##########################################################################################

svg_start = ("""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www."""
         """w3.org/1999/xlink" width="256px" height="256px" style="-ms-trans"""
         """form: rotate(360deg); -webkit-transform: rotate(360deg); transfo"""
         """rm: rotate(360deg);" preserveAspectRatio="xMidYMid meet" viewBox"""
         """="0 0 24 24"><path d=\"""")
svg_end = """\" fill="currentColor"/></svg>"""

COMMAND_RX = re.compile("([MmLlHhVvCcSsQqTtAaZz])")
FLOAT_RX = re.compile("[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

def svg_render(tensor):
    """Converts SVG decoder output into HTML svg."""
    # undo normalization
    tensor = (tensor * stdev_npz) + mean_npz

    # convert to html
    tensor = svg_utils.make_simple_cmds_long(tensor)
    vector = tf.squeeze(tensor, [0, 2])
    html = svg_utils.vector_to_svg(vector.numpy(), stop_at_eos=True, categorical=True)

    # some aesthetic postprocessing
    html = postprocess(html)
    html = html.replace('256px', '50px')

    return html

def svg_html_to_path_string(svg):
    return svg.replace(svg_start, '').replace(svg_end, '')

def _tokenize(pathdef):
    """Returns each svg token from path list."""
    
    # e.g.: 'm0.1-.5c0,6' -> m', '0.1, '-.5', 'c', '0', '6'
    for x in COMMAND_RX.split(pathdef):
        if x != '' and x in 'MmLlHhVvCcSsQqTtAaZz':
            yield x
        for token in FLOAT_RX.findall(x):
            yield token


def path_string_to_tokenized_commands(path):
    """Tokenizes the given path string.

    E.g.:
        Given M 0.5 0.5 l 0.25 0.25 z
        Returns [['M', '0.5', '0.5'], ['l', '0.25', '0.25'], ['z']]
    """
    new_path = []
    current_cmd = []
    for token in _tokenize(path):
        if len(current_cmd) > 0:
            if token in 'MmLlHhVvCcSsQqTtAaZz':
                # cmd ended, convert to vector and add to new_path
                new_path.append(current_cmd)
                current_cmd = [token]
            else:
                # add arg to command
                current_cmd.append(token)
        else:
            # add to start new cmd
            current_cmd.append(token)

    if current_cmd:
        # process command still unprocessed
        new_path.append(current_cmd)

    return new_path


def separate_substructures(tokenized_commands):
    """Returns a list of SVG substructures."""
    # every moveTo command starts a new substructure
    # an SVG substructure is a subpath that closes on itself
    # such as the outter and the inner edge of the character `o`
    substructures = []
    curr = []
    
    for cmd in tokenized_commands:
        if cmd[0] in 'mM' and len(curr) > 0:
            substructures.append(curr)
            curr = []
        curr.append(cmd)
    if len(curr) > 0:
        substructures.append(curr)
    return substructures


def postprocess(svg, dist_thresh=2., skip=False):
    path = svg_html_to_path_string(svg)
    svg_template = svg.replace(path, '{}')
    tokenized_commands = path_string_to_tokenized_commands(path)

    dist = lambda a, b: np.sqrt((float(a[0]) - float(b[0]))**2 + (float(a[1]) - float(b[1]))**2)
    are_close_together = lambda a, b, t: dist(a, b) < t

    # first, go through each start/end point and merge if they're close enough
    # together (that is, make end point the same as the start point).
    # TODO: there are better ways of doing this, in a way that propagates error
    # back (so if total error is 0.2, go through all N commands in this
    # substructure and fix each by 0.2/N (unless they have 0 vertical change))
    substructures = separate_substructures(tokenized_commands)
    previous_substructure_endpoint = (0., 0.,)
  
    for substructure in substructures:
        # first, if the last substructure's endpoint was updated, we must update
        # the start point of this one to reflect the opposite update
        substructure[0][-2] = str(float(substructure[0][-2]) -
                                  previous_substructure_endpoint[0])
        substructure[0][-1] = str(float(substructure[0][-1]) -
                                  previous_substructure_endpoint[1])
    
        start = list(map(float, substructure[0][-2:]))
        curr_pos = (0., 0.)
        for cmd in substructure:
          curr_pos, _ = svg_utils._update_curr_pos(curr_pos, cmd, (0., 0.))
          
        if are_close_together(start, curr_pos, dist_thresh):
            new_point = np.array(start)
            previous_substructure_endpoint = ((new_point[0] - curr_pos[0]), (new_point[1] - curr_pos[1]))
            
            substructure[-1][-2] = str(float(substructure[-1][-2]) + (new_point[0] - curr_pos[0]))
            substructure[-1][-1] = str(float(substructure[-1][-1]) + (new_point[1] - curr_pos[1]))
            
            if substructure[-1][0] in 'cC':
                substructure[-1][-4] = str(float(substructure[-1][-4]) + (new_point[0] - curr_pos[0]))
                substructure[-1][-3] = str(float(substructure[-1][-3]) + (new_point[1] - curr_pos[1]))
      
    if skip:
        return svg_template.format(' '.join([' '.join(' '.join(cmd) for cmd in s) for s in substructures]))
  
    cosa = lambda x, y: (x[0] * y[0] + x[1] * y[1]) / ((np.sqrt(x[0]**2 + x[1]**2) * np.sqrt(y[0]**2 +  y[1]**2)))
    rotate = lambda a, x, y: (x * np.cos(a) - y * np.sin(a), y * np.cos(a) + x * np.sin(a))

    # second, find adjacent bezier curves and, if their control points are almost aligned, 
    # fully align them

    for substructure in substructures:
        curr_pos = (0., 0.)
        new_curr_pos, _ = svg_utils._update_curr_pos((0., 0.,), substructure[0], (0., 0.))
    
        for cmd_idx in range(1, len(substructure)):
            prev_cmd = substructure[cmd_idx-1]
            cmd = substructure[cmd_idx]

            new_new_curr_pos, _ = svg_utils._update_curr_pos(new_curr_pos, cmd, (0., 0.))
      
            if cmd[0] == 'c':
                if prev_cmd[0] == 'c':
                    # check the vectors and update if needed
                    # previous control pt wrt new curr point
                    prev_ctr_point = (curr_pos[0] + float(prev_cmd[3]) - new_curr_pos[0],
                                      curr_pos[1] + float(prev_cmd[4]) - new_curr_pos[1])
                    ctr_point = (float(cmd[1]), float(cmd[2]))

                    if -1. < cosa(prev_ctr_point, ctr_point) < -0.95:
                        # calculate exact angle between the two vectors
                        angle_diff = (np.pi - np.arccos(cosa(prev_ctr_point, ctr_point)))/2

                        # rotate each vector by angle/2 in the correct direction for each.
                        sign = np.sign(np.cross(prev_ctr_point, ctr_point))
                        new_ctr_point = rotate(sign * angle_diff, *ctr_point)
                        new_prev_ctr_point = rotate(-sign * angle_diff, *prev_ctr_point)

                        # override the previous control points
                        # (which has to be wrt previous curr position)
                        substructure[cmd_idx-1][3] = str(new_prev_ctr_point[0] -
                                                         curr_pos[0] + new_curr_pos[0])
                        substructure[cmd_idx-1][4] = str(new_prev_ctr_point[1] -
                                                         curr_pos[1] + new_curr_pos[1])
                        substructure[cmd_idx][1] = str(new_ctr_point[0])
                        substructure[cmd_idx][2] = str(new_ctr_point[1])

        curr_pos = new_curr_pos
        new_curr_pos = new_new_curr_pos
  
    return svg_template.format(' '.join([' '.join(' '.join(cmd) for cmd in s) 
        for s in substructures]))

##########################################################################################    

def test_font_glyph_inference(fontname, glyph, inputt2tpath):
    print(f'{bcolors.BOLD}Testing font glyph inference ({fontname}: {glyph})...{bcolors.ENDC}')
    
    modelbasepath = basepath/'models-google'
    modelsuffix = '_external'
    
    uni = ord(glyph)
    glyphpath = inputt2tpath/f'{fontname}-{uni}'
    
    hparam_set = 'svg_decoder'
    vae_ckpt_dir = os.fspath(modelbasepath/f'image_vae{modelsuffix}')
    add_hparams = f'vae_ckpt_dir={vae_ckpt_dir},vae_hparam_set=image_vae'
                   
    model_name = 'svg_decoder'
    ckpt_dir = os.fspath(modelbasepath/f'svg_decoder{modelsuffix}')

    Path('./out-font-glyph-inference.html').write_text('\n'.join(infer_from_file(
        glyphpath, hparam_set, add_hparams, model_name, ckpt_dir)))

def test_svg_inference():
    print(f'{bcolors.BOLD}Testing SVG inference...{bcolors.ENDC}')

    modelbasepath = basepath/'models-google'
    modelsuffix = '_external'
    
    uni = ord('U')
    svg = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="50px" height="50px" style="-ms-transform: rotate(360deg); -webkit-transform: rotate(360deg); transform: rotate(360deg);" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path d="m 9.46800041199 5.86666679382 l 1.7039999961853027 0.0 l 0.0 15.095999717712402 l 1.656000018119812 0.0 l 0.0 -15.095999717712402 l 1.7039999961853027 0.0 l 0.0 16.799999237060547 l -5.064000129699707 0.0 l 1.1920928955078125e-07 -16.799999237060547" fill="currentColor"/></svg>'
    
    example = generate_t2t_example(uni, svg)

    hparam_set = 'svg_decoder'
    vae_ckpt_dir = os.fspath(modelbasepath/f'image_vae{modelsuffix}')
    add_hparams = f'vae_ckpt_dir={vae_ckpt_dir},vae_hparam_set=image_vae'
                   
    model_name = 'svg_decoder'
    ckpt_dir = os.fspath(modelbasepath/f'svg_decoder{modelsuffix}')

    Path('./out-svg-inference.html').write_text('\n'.join(infer(
        example, hparam_set, add_hparams, model_name, ckpt_dir)))

##########################################################################################    

@app.route('/api')
def index():
    return "Public Foundry Combinator"

##########################################################################################    

# @app.route('/run-tests')
# def run_tests():
#     fontname = FLAGS.font
# 
#     inputpath = inferencepath/fontname/'input'
#     inputt2tpath = inputpath/'t2t'
# 
#     test_font_glyph_inference(fontname, 'C', inputt2tpath)
#     test_svg_inference()
#     
#     return "Tests Complete"

##########################################################################################    

@app.route('/api/fonts')
def get_fonts():

    fontdirs = inferencepath.glob('*/')
    fonts = [f.name for f in fontdirs if f.is_dir()]
    
    return jsonify({ 'fonts': fonts })

##########################################################################################    

@app.route('/api/inputs/<string:fontname>', methods=['GET'])
def get_inputs(fontname):
    # sample: 
    # http://127.0.0.1:5959/api/inputs/Unica?json=False
        
    use_json = request.args.get('json', default='true').lower() == 'true'

    glyphspath = inferencepath/fontname/'input'/'glyphs'
    glyphpaths = glyphspath.glob('*.sfd')

    inputs = {}

    ymin = None
    ymax = None

    print(f'{datetime.datetime.now()}: {bcolors.BOLD}Getting {fontname} inputs as SVGs...{bcolors.ENDC}', end='')
    print(f'{bcolors.OKGREEN}SUCCESS{bcolors.ENDC}')
    
    # preprocess -- get ymin / ymax
    
    for glyphindex, glyphpath in enumerate(glyphpaths):
        uni = int(glyphpath.with_suffix('').name)

        # open the glyph to get the paths
    
        f = fontforge.open(os.fspath(glyphpath))
        
        tempsvgfile = tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False)
        tempsvgfile.close()
                
        f.generate(tempsvgfile.name)
        
        svg = Path(tempsvgfile.name).read_text()
        
        path_obj, ymin, ymax = get_svg_path_ymin_ymax(svg, ymin=ymin, ymax=ymax, tag='glyph')
        inputs[chr(uni)] = path_obj
        
        Path(tempsvgfile.name).unlink()
        f.close()

    # compose

    for i, (key, value) in enumerate(inputs.items()):
        inputs[key] = compose_svg(inputs[key], ymin=ymin, ymax=ymax, flipv=True)
        
    if not use_json:
        return '\n'.join(inputs.values())

    return jsonify({'inputs': inputs})

##########################################################################################    

@app.route('/api/infer/autotrace/<string:modelname>/<string:modelsuffix>/<string:fontname>/<string:glyph>', methods=['GET'])
def infer_autotrace_from_font(modelname, modelsuffix, fontname, glyph):
    # sample: 
    # http://127.0.0.1:5959/api/infer/bitmap/models-google/external/Unica/A?json=False
        
    use_json = request.args.get('json', default='true').lower() == 'true'
    bitmap_depth = 1 # bitmap_depth is always 1
    bitmap_fill = True # always fill outlines

    modelbasepath = basepath/modelname
    modelsuffix = '' if modelsuffix == '-' else f'_{modelsuffix}'

    inputpath = inferencepath/fontname/'input'
    inputt2tpath = inputpath/'t2t'

    uni = ord(glyph)
    glyphpath = inputt2tpath/f'{fontname}-{uni}'
    
    print(f'{datetime.datetime.now()}: {bcolors.BOLD}autotrace/{fontname} "{glyph}" inference using {modelname}{modelsuffix} (use_json: {use_json})...{bcolors.ENDC}', end='')
   
    hparam_set = 'image_vae'
    add_hparams = ''
                   
    model_name = 'image_vae'
    ckpt_dir = os.fspath(modelbasepath/f'image_vae{modelsuffix}')

    inf = infer_from_file(glyphpath, hparam_set, add_hparams, model_name, ckpt_dir, out="PIL_image", bitmap_depth=bitmap_depth, bitmap_fill=bitmap_fill)
    
    print(f'{bcolors.OKGREEN}SUCCESS{bcolors.ENDC}')
    
    inferences = {}

    ymin = None
    ymax = None
    
    # zip up our autotraced inferences with our glyphs
    # preprocess -- get ymin / ymax

    for i, image in enumerate(inf):
        glyph = glyphs[i]
        
        # save to bmp 
        
        tempbitmapfile = tempfile.NamedTemporaryFile(mode='w', suffix='.png', delete=False)
        tempsvgfile = tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False)

        tempbitmapfile.close()
        tempsvgfile.close()
        
        tempbitmappath = Path(tempbitmapfile.name)
        tempsvgpath = Path(tempsvgfile.name)

        image.save(tempbitmappath, format="png")
        
        # create a new sfd
        
        vwidth = font_height
    
        f = fontforge.open('./blank.sfd')
        f.ascent = font_ascent
        f.descent = font_height - font_ascent

        # paste in the bitmap and autotrace
        
        char = f.createChar(uni)
        char.importOutlines(tempbitmapfile.name)
        char.autoTrace()
        char.vwidth = vwidth
    
        f.selection.all()
        f.autoWidth(100, 30)
        f.autoHint()

        # save to svg
    
        f.generate(tempsvgfile.name)
        f.close()
    
        # read svg back in, clean and center
    
        svg = tempsvgpath.read_text()
        
        path_obj, ymin, ymax = get_svg_path_ymin_ymax(svg, ymin=ymin, ymax=ymax, tag='glyph')
        inferences[glyph] = path_obj
        
        # cleanup
                
        tempbitmappath.unlink()
        tempsvgpath.unlink()

    # compose

    for i, (key, value) in enumerate(inferences.items()):
        inferences[key] = compose_svg(inferences[key], ymin=ymin, ymax=ymax, flipv=True)
       
    if not use_json:
        return '\n'.join(inferences.values())
    
    return jsonify({'inferences': inferences})

##########################################################################################    

@app.route('/api/infer/bitmap/<string:modelname>/<string:modelsuffix>/<string:fontname>/<string:glyph>', methods=['GET'])
def infer_bitmap_from_font(modelname, modelsuffix, fontname, glyph):
    # sample: 
    # http://127.0.0.1:5959/api/infer/bitmap/models-google/external/Unica/A?json=False
        
    use_json = request.args.get('json', default='true').lower() == 'true'
    bitmap_depth = int(request.args.get('depth', default='8'))
    bitmap_fill = request.args.get('fill', default='true').lower() == 'true'

    modelbasepath = basepath/modelname
    modelsuffix = '' if modelsuffix == '-' else f'_{modelsuffix}'

    inputpath = inferencepath/fontname/'input'
    inputt2tpath = inputpath/'t2t'

    uni = str(ord(glyph))
    glyphpath = inputt2tpath/f'{fontname}-{uni}'
    
    print(f'{datetime.datetime.now()}: {bcolors.BOLD}bitmap/{fontname} "{glyph}" inference using {modelname}{modelsuffix} (use_json: {use_json}, bitmap_depth: {bitmap_depth}-bit, bitmap_fill: {bitmap_fill})...{bcolors.ENDC}', end='')
   
    hparam_set = 'image_vae'
    add_hparams = ''
                   
    model_name = 'image_vae'
    ckpt_dir = os.fspath(modelbasepath/f'image_vae{modelsuffix}')

    inf = infer_from_file(glyphpath, hparam_set, add_hparams, model_name, ckpt_dir, out="img", bitmap_depth=bitmap_depth, bitmap_fill=bitmap_fill)
    
    print(f'{bcolors.OKGREEN}SUCCESS{bcolors.ENDC}')
    
    inferences = {}
    
    # zip up our bitmap inferences with our glyphs

    for i, img in enumerate(inf):
        glyph = glyphs[i]
        inferences[glyph] = img
        
    if not use_json:
        return '\n'.join(inferences.values())
    
    return jsonify({'inferences': inferences})

##########################################################################################    

@app.route('/api/infer/svg/<string:modelname>/<string:modelsuffix>/<string:fontname>/<string:glyph>', methods=['GET'])
def infer_svg_from_font(modelname, modelsuffix, fontname, glyph):
    # sample: 
    # http://127.0.0.1:5959/api/infer/svg/models-google/external/Unica/A?json=False
        
    use_json = request.args.get('json', default='true').lower() == 'true'

    modelbasepath = basepath/modelname
    modelsuffix = '' if modelsuffix == '-' else f'_{modelsuffix}'

    inputpath = inferencepath/fontname/'input'
    inputt2tpath = inputpath/'t2t'

    uni = ord(glyph)
    glyphpath = inputt2tpath/f'{fontname}-{uni}'
    
    print(f'{datetime.datetime.now()}: {bcolors.BOLD}svg/{fontname} "{glyph}" inference using {modelname}{modelsuffix} (use_json: {use_json})...{bcolors.ENDC}', end='')
   
    hparam_set = 'svg_decoder'
    vae_ckpt_dir = os.fspath(modelbasepath/f'image_vae{modelsuffix}')
    add_hparams = f'vae_ckpt_dir={vae_ckpt_dir},vae_hparam_set=image_vae'
                   
    model_name = 'svg_decoder'
    ckpt_dir = os.fspath(modelbasepath/f'svg_decoder{modelsuffix}')

    inf = infer_from_file(glyphpath, hparam_set, add_hparams, model_name, ckpt_dir)
    
    print(f'{bcolors.OKGREEN}SUCCESS{bcolors.ENDC}')
    
    inferences = {}

    ymin = None
    ymax = None
    
    # clean our svg, zip up our inferences with our glyphs
    # preprocess -- get ymin / ymax

    for i, svg in enumerate(inf):
        glyph = glyphs[i]

        path_obj, ymin, ymax = get_svg_path_ymin_ymax(svg, ymin=ymin, ymax=ymax)
        inferences[glyph] = path_obj

    # compose

    for i, (key, value) in enumerate(inferences.items()):
        inferences[key] = compose_svg(inferences[key], ymin=ymin, ymax=ymax)
        
    if not use_json:
        return '\n'.join(inferences.values())
    
    return jsonify({'inferences': inferences})

##########################################################################################    

@app.route('/api/infer/svg/<string:modelname>/<string:modelsuffix>/<string:glyph>', methods=['GET', 'POST'])
def infer_svg_from_svg(modelname, modelsuffix, glyph):
    # sample: 
    # http://127.0.0.1:5959/api/infer/svg/models-google/external/U?json=False&svg=%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20xmlns%3Axlink%3D%22http%3A%2F%2Fwww.w3.org%2F1999%2Fxlink%22%20width%3D%2250px%22%20height%3D%2250px%22%20style%3D%22-ms-transform%3A%20rotate%28360deg%29%3B%20-webkit-transform%3A%20rotate%28360deg%29%3B%20transform%3A%20rotate%28360deg%29%3B%22%20preserveAspectRatio%3D%22xMidYMid%20meet%22%20viewBox%3D%220%200%2024%2024%22%3E%3Cpath%20d%3D%22m%209.46800041199%205.86666679382%20l%201.7039999961853027%200.0%20l%200.0%2015.095999717712402%20l%201.656000018119812%200.0%20l%200.0%20-15.095999717712402%20l%201.7039999961853027%200.0%20l%200.0%2016.799999237060547%20l%20-5.064000129699707%200.0%20l%201.1920928955078125e-07%20-16.799999237060547%22%20fill%3D%22currentColor%22%2F%3E%3C%2Fsvg%3E
    
    svg = None
    
    if request.method == 'POST':
        if request.json and 'svg' in request.json:
            svg = request.json['svg']
    else:
        svg = request.args.get('svg')

    if not svg:
        return jsonify({ 'error': "Request must include 'svg'" })
    
    use_json = request.args.get('json', default='true').lower() == 'true'
    
    modelbasepath = basepath/modelname
    modelsuffix = '' if modelsuffix == '-' else f'_{modelsuffix}'

    uni = ord(glyph)
    print(f'{datetime.datetime.now()}: ', end='')
    result = generate_t2t_example(uni, svg)
        
    if result['error']:
        return jsonify({ 'error': result['error'] })
    
    if not result['example']:
        return jsonify({ 'error': "'generate_t2t_example' could not produce an example" })
        
    print(f'{datetime.datetime.now()}: {bcolors.BOLD}svg/SVG glyph inference "{glyph}" using {modelname}{modelsuffix} (use_json: {use_json})...{bcolors.ENDC}', end='')
    example = result['example']

    hparam_set = 'svg_decoder'
    vae_ckpt_dir = os.fspath(modelbasepath/f'image_vae{modelsuffix}')
    add_hparams = f'vae_ckpt_dir={vae_ckpt_dir},vae_hparam_set=image_vae'
                   
    model_name = 'svg_decoder'
    ckpt_dir = os.fspath(modelbasepath/f'svg_decoder{modelsuffix}')

    inf = infer(example, hparam_set, add_hparams, model_name, ckpt_dir)
    
    print(f'{bcolors.OKGREEN}SUCCESS{bcolors.ENDC}')

    inferences = {}

    ymin = None
    ymax = None
        
    # clean our svg, zip up our inferences with our glyphs
    # preprocess -- get ymin / ymax

    for i, svg in enumerate(inf):
        glyph = glyphs[i]

        path_obj, ymin, ymax = get_svg_path_ymin_ymax(svg, ymin=ymin, ymax=ymax)
        inferences[glyph] = path_obj
        
    # compose

    for i, (key, value) in enumerate(inferences.items()):
        inferences[key] = compose_svg(inferences[key], ymin=ymin, ymax=ymax)

    if not use_json:
        return '\n'.join(inferences.values())
    
    return jsonify({'inferences': inferences})

##########################################################################################    

def main(_):
    # assume we've run generate-font-inference-dataset.py on each of our 
    # possible inference fonts
    
    print(f'{datetime.datetime.now()}: {bcolors.BOLD}Starting server{bcolors.ENDC}')
    socketio.run(app, port=FLAGS.port, host=FLAGS.host)
    
if __name__ == '__main__':
    FLAGS = flags.FLAGS
    
    flags.DEFINE_string('font', 'Unica', 'Font to use for test inference.')
    flags.DEFINE_integer('port', 5959, 'Port to listen on.')
    flags.DEFINE_string('host', '0.0.0.0', 'The hostname or IP to listen on.')

    absl.app.run(main)
