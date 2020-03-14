#!/home/krister/anaconda3/envs/public-foundry/bin/python

import sys, os, tempfile
from pathlib import Path
from pprint import PrettyPrinter

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
from tensor2tensor.data_generators import generator_utils

import fontforge
import svg_utils

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
# HERE WE GO
##########################################################################################

maxpaths = 50

pp = PrettyPrinter(compact=True)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # shut up and play the hits

# borrowed from glyphtracer

font_ascent = 1000
font_height = 780

##########################################################################################
# TFRECORD / T2T GENERATION
#
# https://tensorflow.github.io/tensor2tensor/new_problem.html
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/data_generators/problem.py
# https://tensorflow.github.io/tensor2tensor/overview.html
##########################################################################################

def _parse_svg(uni, svg):
    """Leverage FontForge to convert our svg -> path"""
    
    tempsvgfile = tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False)
    tempsfdfile = tempfile.NamedTemporaryFile(suffix='.sfd', delete=False)

    tempsvgfile.write(svg)
    tempsvgfile.close()
    tempsfdfile.close()

    # this is our basic procedure for creating a typeface from an svg / multiple svg files
    
    vwidth = font_height
    
    f = fontforge.open('./blank.sfd')
    f.ascent = font_ascent
    f.descent = font_height - font_ascent

    char = f.createChar(uni)
    char.importOutlines(tempsvgfile.name)
    f[uni].vwidth = vwidth

    # all our glyphs are in
    
    f.selection.all()
    f.autoWidth(100, 30)
    f.autoHint()

    width = f[uni].width
    
    f.save(tempsfdfile.name)
    f.close()
    
    sfd = Path(tempsfdfile.name).read_text()
    g = {
        'width': width, 
        'vwidth': vwidth, 
        'sfd': sfd,
    }

    path = svg_utils.sfd_to_path_list(g)
    path = svg_utils.add_missing_cmds(path, remove_zs=False)
    path = svg_utils.normalize_based_on_viewbox(path, '0 0 {} {}'.format(g['width'], g['vwidth']))

    # add path optimization here

    # cleanup
    
    Path(tempsvgfile.name).unlink()
    Path(tempsfdfile.name).unlink()
    
    return path, width, vwidth

def _is_valid_glyph(uni, width, vwidth):
    """Checks validity of glyph."""
   
    is_09 = 48 <= uni <= 57
    is_capital_az = 65 <= uni <= 90
    is_az = 97 <= uni <= 122
    is_valid_dims = width != 0 and vwidth != 0

    return (is_09 or is_capital_az or is_az) and is_valid_dims

def _create_example(uni, path):
    """Bulk of dataset processing. Converts str path to serialized tf.Example."""

    final = {}

    # zoom out
    path = svg_utils.zoom_out(path)
    # make clockwise
    path = svg_utils.canonicalize(path)

    # render path for training
    final['rendered'] = svg_utils.per_step_render(path, absolute=True)

    # make path relative
    path = svg_utils.make_relative(path)
    # convert to vector
    vector = svg_utils.path_to_vector(path, categorical=True)
    # make simple vector
    vector = np.array(vector)
    vector = np.concatenate(
        [np.take(vector, [0, 4, 5, 9], axis=-1), vector[..., -6:]], axis=-1)

    # count some stats
    final['seq_len'] = np.shape(vector)[0]
    final['class'] = int(svg_utils.map_uni_to_alphanum(uni))

    # append eos
    vector = svg_utils.append_eos(vector.tolist(), True, 10)

    # pad path to 51 (with eos)
    final['sequence'] = np.concatenate(
        (vector, np.zeros(((50 - final['seq_len']), 10))), 0)

    # make pure list:
    final['rendered'] = np.reshape(final['rendered'][..., 0],
        [64*64]).astype(np.float32).tolist()
    final['sequence'] = np.reshape(final['sequence'],
        [51*10]).astype(np.float32).tolist()
    final['class'] = np.reshape(final['class'],
        [1]).astype(np.int64).tolist()
    final['seq_len'] = np.reshape(final['seq_len'],
        [1]).astype(np.int64).tolist()

    return generator_utils.to_example(final)

def _generate_sample(example):
    """Generate sample of target svg commands."""
     
    yield {
        'targets_sln': np.array(
            example.features.feature['seq_len'].int64_list.value).astype(
                np.int64).tolist(),
        'targets_cls': np.array(
            example.features.feature['class'].int64_list.value).astype(
                np.int64).tolist(),
        'targets_rel': np.array(
            example.features.feature['sequence'].float_list.value).astype(
                np.float32).tolist(),
        'targets_rnd': np.array(
            example.features.feature['rendered'].float_list.value).astype(
                np.float32).tolist()
    }

def generate_t2t_example(uni, svg):
    print(f'{bcolors.BOLD}Generating tfrecord...{bcolors.ENDC}', end = '')

    path, width, vwidth = _parse_svg(uni, svg)
    errorString = None

    if _is_valid_glyph(uni, width, vwidth):        
        if len(path) > maxpaths:
            # too many paths!
            
            errorString = f'{chr(uni)} ({uni}) has too many paths: {len(path)}'
        elif len(path) == 0: 
            # no paths!
            
             errorString = f'{chr(uni)} ({uni}) has no paths'
        else:
            # super clunky but we have to get our example in the right format
            
            example = _create_example(uni, path)
            
            tempexamplefile = tempfile.NamedTemporaryFile(mode='w', delete=False)
            tempexamplefile.close()
            Path(tempexamplefile.name).unlink() # we must delete before we generate_files

            generator_utils.generate_files(_generate_sample(example), [ tempexamplefile.name ], max_cases=1)
            
            # https://www.tensorflow.org/tutorials/load_data/tfrecord
            raw_dataset = tf.data.TFRecordDataset([ tempexamplefile.name ])

            for raw_record in raw_dataset.take(1):
                example = raw_record
            
            Path(tempexamplefile.name).unlink() # delete for real
            
            print(f'{bcolors.OKGREEN}SUCCESS{bcolors.ENDC}')
            return { 'error': None, 'example': example }
    else:
        errorString = f'{chr(uni)} ({uni}) is invalid'
        
    print(f'{bcolors.FAIL}{errorString}{bcolors.ENDC}')
    return { 'error': errorString, 'example': None }
   
##########################################################################################    

def main(_):

    # test svg (U): 
    # <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="50px" height="50px" style="-ms-transform: rotate(360deg); -webkit-transform: rotate(360deg); transform: rotate(360deg);" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path d="m 9.46800041199 5.86666679382 l 1.7039999961853027 0.0 l 0.0 15.095999717712402 l 1.656000018119812 0.0 l 0.0 -15.095999717712402 l 1.7039999961853027 0.0 l 0.0 16.799999237060547 l -5.064000129699707 0.0 l 1.1920928955078125e-07 -16.799999237060547" fill="currentColor"/></svg>
    glyph = FLAGS.glyph
    svg = FLAGS.svg
    uni = ord(glyph)
        
    # generate our t2t example
    result = generate_t2t_example(uni, svg)
    
    if FLAGS.debug: 
        pp.pprint(result)

if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('svg', '', 'SVG string for inference.')
    flags.DEFINE_string('glyph', '', 'Glyph for inference.')
    flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
    
    tf.compat.v1.enable_eager_execution() # necessary for generate_t2t_example 

    app.run(main)
