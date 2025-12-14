%% 1. データの読み込み
clear; clc; close all;
load('result.mat'); % t, label, ref (スコア), y_lstm などが入っています

%% 2. 攻撃開始点（立ち上がり）の自動検出
% labelの差分をとり、正の値(0->1)になっているインデックスを探す
attack_starts = find(diff(label) > 0);

if isempty(attack_starts)
    error('攻撃ラベル(1)が見つかりませんでした。データを確認してください。');
end

% 最初の攻撃に注目する
first_attack_idx = attack_starts(1);
fprintf('最初の攻撃開始インデックス: %d (時刻: %.2f秒)\n', first_attack_idx, t(first_attack_idx));

%% 3. 表示範囲の設定（攻撃の前後 ±500サンプル）
window_size = 500; 

% 配列の範囲外に出ないように調整
start_idx = max(1, first_attack_idx - window_size);
end_idx   = min(length(label), first_attack_idx + window_size);
range_indices = start_idx:end_idx;

%% 4. 拡大プロット
figure('Name', 'Attack Zoom Inspector', 'Color', 'w');

% 上段: 正解ラベル (Ground Truth)
subplot(2,1,1);
plot(t(range_indices), label(range_indices), 'r-', 'LineWidth', 2);
title('正解ラベル (Ground Truth): 0=正常, 1=攻撃');
grid on;
ylim([-0.2, 1.2]); % 0と1が見やすいように
ylabel('Label');

% 下段: 異常スコア (Anomaly Score)
subplot(2,1,2);
plot(t(range_indices), ref(range_indices), 'b-', 'LineWidth', 1.5);
hold on;
% 攻撃開始位置に縦線を引いてわかりやすくする
xline(t(first_attack_idx), 'k--', 'Attack Start'); 
title('異常検知スコア (Blue Line)');
grid on;
ylabel('Score');
xlabel('Time [sec]');

% ※学習不足で値がおかしいとのことですが、
%   ここで「ラベルが上がった瞬間に、青線が少しでも反応しているか（遅れているか）」
%   を確認できれば、実験環境としては成功です。