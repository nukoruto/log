%% 1. 初期化とデータのロード
clear; clc;

% パイプラインが出力した .mat ファイルを読み込む
% ※パスは実際のファイルの場所に合わせて変更してください
load('result.mat'); 

% 読み込まれる変数の確認:
% t      : 時間軸 [秒]
% label  : 正解データ (0=正常, 1=攻撃)
% ref    : LSTMの総合異常スコア (S)
% y_lstm : LSTMの時間異常スコア (s_time)
% y_pid  : 比較用のPID制御出力 (ある場合)

%% 2. Simulink用データ形式 (timeseries) への変換
% ベクトルデータのままだと扱いづらいため、時間軸とセットのオブジェクトに変換します。

% (1) 正解ラベル (Ground Truth)
% 名前を 'GroundTruth' に設定しておくとScopeで分かりやすいです
ts_label = timeseries(label, t);
ts_label.Name = 'GroundTruth';
ts_label.DataInfo.Interpolation = tsdata.interpolation('zoh'); % 0次ホールド(矩形波)

% (2) LSTM総合異常スコア (Anomaly Score)
% 変数 'ref' に総合スコア(S)が入っています
ts_score = timeseries(ref, t);
ts_score.Name = 'LSTM_Score_S';
ts_score.DataInfo.Interpolation = tsdata.interpolation('linear'); % 線形補間(滑らか)

% (3) (任意) 時間スコア成分のみを見たい場合
ts_time_score = timeseries(y_lstm, t);
ts_time_score.Name = 'LSTM_Time_Component';

%% 3. シミュレーション設定の自動化
% データの終了時刻をシミュレーション終了時間にセットするための変数
SimStopTime = t(end);

fprintf('データのロード完了。\n');
fprintf('シミュレーション時間: %.2f 秒\n', SimStopTime);
fprintf('Simulinkモデルの "終了時間 (Stop Time)" に変数名 SimStopTime と入力してください。\n');