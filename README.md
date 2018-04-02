# eeg basic autoencoder

This time series encoder was adapted form the pytorch sequence to sequence encoder tutorials and then reduced to a basic encoder.
http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

### Data
Data used for this encoder can be found here:
https://physionet.org/pn6/capslpdb/

### Install
To use this code please install pytorch, and mne

http://pytorch.org

MNE can be installed alongside other useful tools by installing braindecode

https://robintibor.github.io/braindecode/index.html

Download a copy of the python tsne implimentation and add the file to the same directory as `ae.py` to do tsne analysis

https://lvdmaaten.github.io/tsne/

### Run

Running the program with the following command will run with some default parameters, edit these in file if you want to change them.
```
python -i ae.py
```
To train an encoder, use the following command
```
pairs = make_pairs(data, LENGTH_PER_CHANNEL)
encoder1, decoder1, criterion1 = setup_encoder_decoder(hidden_size, MAX_LENGTH)
losses = run_train_iters_v2(encoder1, decoder1, criterion1,pairs, hidden_size, MAX_LENGTH, repeat_over_data_times)
```
To show the graph of losses over time use
```
showPlot(losses)
```

### t-SNE analysis

To run analysis you can follow this example, or experiment with your own values. See the tsne implimentation code for more information.
```
# Reduce the number of dims from 45 to 2
Y = tsne(X, 2, 45, 20.0)
# Generate the scatter plot based off the generated Y values
pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
# Show the scatter plot
pylab.show()
```
