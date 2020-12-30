# public-foundry-combinator

## About Public Foundry

[Public Foundry](http://publicfoundry.ai) is a project by Yvan Martinez, Joshua Trees and Krister Olsson, supported by a Google Focused Research Award from the Artists and Machine Intelligence program and the Hoffmitz Milken Center for Typography at ArtCenter College of Design.

## About Public Foundry Combinator

The Public Foundry Combinator was built to test different machine learning models for font generation. This version was built specifically to work with Google's SVG-VAE model architecture: [SVG VAE: Generating Scalable Vector Graphics Typography](https://magenta.tensorflow.org/svg-vae)

The Combinator comprises a Flask-based [server](https://github.com/kolsson/public-foundry-combinator) component and React-based [client](https://github.com/kolsson/public-foundry-combinator-ui) component.

To begin, the user selects a model and a preprocessed font to use for inference. Next, the user can adjust the output format: SVG, inferred using the SVG decoder; Bitmap, inferred using the VAE; and Bitmap to SVG, inferred using the VAE and then postprocessed with potrace to create SVG glyphs. For the latter two output options bitmap depth and / or bitmap contrast can also be adjusted.

Once inference is complete the user can opt to ‘keep’ individual inferred glyphs to build their output typeface. An output typeface can include glyphs from any number of inferences across any number of models.

![Combinator](http://publicfoundry.ai/assets/combinator.png)

## Setup

1. Create a conda environment
```
conda create --name public-foundry-combinator python=3.7 pyqt numpy tensorflow-gpu=1.15 gunicorn flask flask-cors lxml pillow
conda activate public-foundry-combinator
```    
2. Install other dependencies
```
pip install flask_socketio bs4 svgpathtools
pip install tensor2tensor
pip install magenta
```
3. Install [FontForge](https://github.com/fontforge/fontforge/releases/tag/20190801) 

Copy fontforge.so and psMat.so from the FontForge installation to the Combinator base directory

4. Install potrace

**Linux**
```
sudo apt install potrace
```
**Mac**
```
brew install potrace
```

5. Copy SVG-VAE models to the Combinator base directory

Models are have a specific directory structure. The Google "External" model has the following structure:
```
models-google
    svg_decoded_external
        model.ckpt-300000.meta
        model.ckpt-300000.index
        model.ckpt-300000.data-00001-of-00002
        model.ckpt-300000.data-00000-of-00002
        checkpoint
    image_vae_external
        model.ckpt-100000.meta
        model.ckpt-100000.index
        model.ckpt-100000.data-00001-of-00002
        model.ckpt-100000.data-00000-of-00002
        checkpoint
```      
Google model checkpoints can be downloaded [here](https://storage.googleapis.com/magentadata/models/svg_vae/svg_vae.tar.gz).
Models can also be trained following he instructions [here](https://github.com/magenta/magenta/tree/master/magenta/models/svg_vae) and added to the Combinator.

6. The **inference** directory 

The **inference** directory contains a number of preprocessed characters from various scanned typefaces. These typefaces can be selected in the Combinator UI for experimentation. 

Users can add their own preprocessed characters for inference. Details forthcoming.

## Server

To start the server locally:
```
gunicorn --bind 0.0.0.0:5000 wsgi:app
```

The API endpoints documented in the source code can be accessed directly, but the server is most useful when used in concert with the UI [client](https://github.com/kolsson/public-foundry-combinator-ui)

## License

This project is open sourced under MIT license.

This project uses source code from https://github.com/magenta/magenta/tree/master/magenta/models/svg_vae, copyright Google 2019, licensed under the Apache 2.0 license.
