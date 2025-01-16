## About
This repo contains the official MDST implementation of DeepCFR[^1] written during the fall 24 semester, with the implementation transfered to the OpenSpiel[^2] environment. 

## Installations
In order to train and play against the model, install:
```bash
pip install -r requirements.txt
```
Openspiel provides libraries that calculate the exploitabilty of a policy. To run these, first execute the script to install the proper libraries:
```bash
./install_libs.sh
``` 

## Usage
To train the model run the command
```python
python3 train_model.py --iterations 10 --K 5 --save_path "./cfr_model.pth" 
```
Which will train the DeepCFR model for 10 iterations per player, with 5 tree traversals per iteration, saving the final model into the file "cfr_model.pth". We trained this model for 300 iterations with 100 traversals per iterations, the resulting weights are provided in the file "cfr_model(300).pth". 

To calculate the exploitability of a model, run:
```python
python3 calculate_exploitability.py --model "path_to_model"
```
This will calcuate the exploitability of whatever model you trained. Note that this is incredily memory intensive, and will most likley OOM crash if just ran on your local device. 

If you want to play against the model, run
```python
python3 play_model.py --opponent_model <path_to_model> --num_hands <num_hands>
```
By default, this will play just 10 hands against the model we've already trained

## References
[^1]: https://arxiv.org/pdf/1811.00164
[^2]: https://github.com/google-deepmind/open_spiel/tree/master