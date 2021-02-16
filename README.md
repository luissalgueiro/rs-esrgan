# RS-ESRGAN: Super-Resolution of Sentinel-2 Imagery Using Generative Adversarial Networks


See our project website [here](https://luissalgueiro.github.io/rs-esrgan/).

We used BasicSR codes, which can be found [here](https://github.com/xinntao/BasicSR), and modified it to suit our purposes to work with remote-sensing data.

## License

This code cannot be used for commercial purposes. Please contact the authors if interested in licensing this software.

## Data
To train with [WorldView-2 European Cities dataset](https://earth.esa.int/eogateway/catalog/worldview-2-european-cities), you must download the data and follow the directions in options/train folder.

To work with WorldView-Sentinel pairs, please contact the authors.

## Training

- Train the model for Super-resolution with European-cities Dataset  ```python train_esrgan_EUCities.py -opt ./option/train/train_ESRGAN_ESRGAN_WV_5x_v1.json ```. Checkpoints and logs will be saved as defined in the json file.


## Contact

For questions and suggestions use the issues section or send an e-mail to luis.fernando.salgueiro@upc.edu