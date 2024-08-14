# lstm-stock-predictor

This is PyTorch implementation of the LSTM(Long-short-term memory) model to predict prices of day-to-day stock data.

# How to run
- Use Python 3.10.9
- Git clone using `git clone https://github.com/ChaitanyaKatti/lstm-stock-predictor.git`
- Install requirements using `pip install -r requirements.txt`
- Download stock data and place the .csv files in `/data/stocks` folder
- Edit data, model, training, and simulation parameters in `config.py`.
- Run main.py using `python main.py`
- Look at the terminal for training progress and the `/plots` folder for results.

# Model Architecture

This architecture allows for 
- Different input(x) and output(y) features
- Different input and output lengths
- The long-term memory state ('c') to be maintained across both LSTMs

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/ChaitanyaKatti/lstm-stock-predictor/assets/96473570/8de3e718-3986-4605-a851-0643837626f0" alt="LSTM Basic" width=70%>
    </td>
  </tr>
  <tr>
    <td align="center"><b>Figure 1:</b> Forward pass in a standard LSTM</td>
  </tr>
  
  <tr>
    <td align="center">
      <img src="https://github.com/ChaitanyaKatti/lstm-stock-predictor/assets/96473570/7e6f9814-2b17-439b-865f-3c5891db411e" alt="LSTM Recurrent Diagram" width=70%>
    </td>
  </tr>
  <tr>
    <td align="center"><b>Figure 2:</b> Recurrency input in LSTM</td>
  </tr>
  
  <tr>
    <td align="center">
      <img src="https://github.com/ChaitanyaKatti/lstm-stock-predictor/assets/96473570/8c393962-0206-465e-a4b8-7cfc4580de74" alt="LSTM Architecture Design" width=95%>
    </td>
  </tr>
  <tr>
    <td align="center"><b>Figure 3:</b> LSTM Architecture in this implementation (Note: Both x and y differ in size and so does input and output length)</td>
  </tr>
</table>

# Results
<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/ChaitanyaKatti/lstm-stock-predictor/assets/96473570/c3adc43c-dd8f-435f-ad10-cd06d47dab23" alt="Loss Curve" width=70%>
    </td>
  </tr>
  <tr>
    <td align="center"><b>Figure 4:</b> Loss Curve over training</td>
  </tr>

  <tr>
    <td align="center">
      <img src="https://github.com/ChaitanyaKatti/lstm-stock-predictor/assets/96473570/7c063445-1558-4723-b360-da6e3bd132ff" alt="Predictions" width=70%>
    </td>
  </tr>
  <tr>
    <td align="center"><b>Figure 5:</b> Prediction </td>
  </tr>

  <tr>
    <td align="center">
      <img src="https://github.com/ChaitanyaKatti/lstm-stock-predictor/assets/96473570/752bdacc-6535-448f-9404-fbce0fef8ffa" alt="Simulation Results" width=70%>
    </td>
  </tr>
  <tr>
    <td align="center"><b>Figure 6:</b> Market simulation results </td>
  </tr>
</table>
