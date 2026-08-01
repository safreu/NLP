# Transformer experiment comparison

| experiment | architecture | reconstruction_weight | activation | position_encoding | heads | head_dim | parameters | test_loss | sari | bert_f1 | rouge_l | flesch_kincaid_grade | entity_preservation | number_preservation | training_duration_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0_historical_baseline | historical_original | 0.0000 | relu | learned | 4 | 32 | 5476995 | 7.5161 | 20.6740 | 0.8239 | 0.1534 | 3.9090 | NA | NA | 43.7401 |
| E1_corrected_baseline | corrected | 0.0000 | relu | learned | 8 | 32 | 11402698 | 7.5497 | 20.7426 | 0.8316 | 0.1518 | 4.2814 | 0.0317 | NA | 75.0210 |
| E2_dual_decoder_lambda_025 | dual_decoder | 0.2500 | relu | learned | 8 | 32 | 18811284 | 9.4278 | 20.9305 | 0.8114 | 0.1403 | 10.4206 | 0.0423 | NA | 127.0864 |
| E3_dual_decoder_lambda_050 | dual_decoder | 0.5000 | relu | learned | 8 | 32 | 18811284 | 11.2398 | 20.9302 | 0.8122 | 0.1394 | 9.7634 | 0.0370 | NA | 125.7400 |
| E4_dual_decoder_lambda_100 | dual_decoder | 1.0000 | relu | learned | 8 | 32 | 18811284 | 14.7995 | 20.9681 | 0.8154 | 0.1416 | 9.1391 | 0.0370 | NA | 127.5988 |
| E5_dual_decoder_swiglu | dual_decoder | 1.0000 | swiglu | learned | 8 | 32 | 19600788 | 14.7144 | 20.7668 | 0.8238 | 0.1503 | 8.4192 | 0.0370 | NA | 138.9883 |
| E6_dual_decoder_rope | dual_decoder | 1.0000 | relu | rope | 8 | 32 | 18614676 | 11.6551 | 21.0348 | 0.8075 | 0.1293 | 17.4107 | 0.0370 | NA | 150.2113 |
| E7_dual_decoder_heads_4 | dual_decoder | 1.0000 | relu | learned | 4 | 64 | 18903444 | 13.7436 | 20.9468 | 0.8264 | 0.1454 | 7.8628 | 0.0317 | NA | 125.6077 |
| E8_dual_decoder_heads_16 | dual_decoder | 1.0000 | relu | learned | 16 | 16 | 18788244 | 14.8637 | 20.7258 | 0.8299 | 0.1542 | 5.6412 | 0.0423 | NA | 138.7596 |
| E9_final_combined | dual_decoder | 1.0000 | relu | learned | 16 | 16 | 18788244 | 14.8637 | 20.7258 | 0.8299 | 0.1542 | 5.6412 | 0.0423 | NA | 134.3438 |
