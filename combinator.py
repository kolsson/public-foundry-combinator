#!/home/krister/anaconda3/envs/public-foundry/bin/python

import sys, os, tempfile
from pathlib import Path
import copy
import re
import warnings
from pprint import PrettyPrinter

from absl import app
from absl import flags
import numpy as np
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

maxpaths = 50

pp = PrettyPrinter(compact=True)

# shut up and play the hits
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# warnings.simplefilter("ignore")

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

def infer_from_bottleneck(features, bottleneck, model):
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

    return model.infer(features, decode_length=0)

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

def infer_from_file(example_file, hparam_set, add_hparams, model_name, ckpt_dir):
    """ Load, decode and infer our example """
    
    # https://www.tensorflow.org/tutorials/load_data/tfrecord
    raw_dataset = tf.data.TFRecordDataset([ os.fspath(example_file) ])

    for raw_record in raw_dataset.take(1):
        example = raw_record # we decode_example in infer()

    return infer(example, hparam_set, add_hparams, model_name, ckpt_dir)

def infer(example, hparam_set, add_hparams, model_name, ckpt_dir):
    """Decodes one example of each class, conditioned on the example."""

    # initialize with t2t data
    # model, hparams = initialize_model_with_t2t(hparam_set, add_hparams, model_name, ckpt_dir)
    # features1 = preprocess_example(example, hparams) # passed by reference

    # OR initialize with example 
    model, hparams, features1 = initialize_model_with_example(decode_example(example), hparam_set, add_hparams, model_name, ckpt_dir)

    # == the number of glyphs
    num_classes = hparams.num_categories 

    # get bottleneck of the features we selected before
    bottleneck1 = get_bottleneck(features1, model)
    bottleneck1 = tf.tile(bottleneck1, [num_classes, 1])

    # create class batch
    new_features = copy.copy(features1)

    clss_batch = tf.reshape([tf.constant([[clss]], dtype=tf.int64) for clss in range(num_classes)], [-1, 1])
    new_features['targets_cls'] = clss_batch

    new_features = _tile(new_features, 'targets_psr', [num_classes, 1, 1])

    inp_target_dim = [num_classes, 1, 1, 1]

    new_features = _tile(new_features, 'inputs', inp_target_dim)
    new_features = _tile(new_features, 'targets', inp_target_dim)

    # run model
    output_batch = infer_from_bottleneck(new_features, bottleneck1, model)

    # render outputs to svg
    # (our inference example is features1['inputs'])
    output_batch = output_batch['outputs']

    svg_list = []    
    for output in tf.split(output_batch, num_classes):
        svg_list.append(svg_render(output))

    return svg_list

##########################################################################################
# SVG
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

    # second,find adjacent bezier curves and, if their control points are almost aligned, 
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
  
    return svg_template.format(' '.join([' '.join(' '.join(cmd) for cmd in s) for s in substructures]))

##########################################################################################    

def test_font_glyph_inference(fontname, inputt2tpath):
    print(f'{bcolors.BOLD}Testing font glyph inference...{bcolors.ENDC}')
    
    modelbasepath = basepath/'models-google'
    modelsuffix = '_external'
    
    uni = str(ord('C'))
    glyphpath = inputt2tpath/f'{fontname}-{uni}'
    
    hparam_set = 'svg_decoder'
    vae_ckpt_dir = os.fspath(modelbasepath/f'image_vae{modelsuffix}')
    add_hparams = f'vae_ckpt_dir={vae_ckpt_dir},vae_hparam_set=image_vae'
                   
    model_name = 'svg_decoder'
    ckpt_dir = os.fspath(modelbasepath/f'svg_decoder{modelsuffix}')

    Path('./out-font-glyph-inference.html').write_text('\n'.join(infer_from_file(glyphpath, hparam_set, add_hparams, model_name, ckpt_dir)))

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

    Path('./out-svg-inference.html').write_text('\n'.join(infer(example, hparam_set, add_hparams, model_name, ckpt_dir)))

##########################################################################################    

def main(_):
    fontname = FLAGS.font

    # assume we've run generate-font-inference-dataset.poy on each of our 
    # possible inference fonts
    
    inputpath = inferencepath/fontname/'input'
    inputt2tpath = inputpath/'t2t'
    
    # let's do a simple test    
    test_font_glyph_inference(fontname, inputt2tpath)
    test_svg_inference()
    
if __name__ == '__main__':
    FLAGS = flags.FLAGS
    flags.DEFINE_string('font', 'Unica', 'Font to use for inference.')
    flags.DEFINE_boolean('debug', False, 'Produces debugging output.')

    app.run(main)