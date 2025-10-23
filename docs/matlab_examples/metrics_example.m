%% metrics_example.m
% Step response metrics example for IAE/ISE/ITAE/Tr/Ts/OS/ESS calculations.
% このサンプルは LSTM 推定値と PID 制御値を比較する最小例です。

% 時間軸と基準信号（ステップ入力）
t = (0:0.05:10)';
ref = ones(size(t));

% サンプル応答（LSTM と PID を想定した時系列）
y_lstm = 1 - exp(-0.8 * t) .* (1 + 0.05 * sin(2.5 * t));
y_pid = 1 - exp(-1.2 * t) + 0.05 * exp(-0.4 * t) .* sin(1.5 * t);

% 指標の計算
metrics_lstm = compute_metrics(t, ref, y_lstm);
metrics_pid = compute_metrics(t, ref, y_pid);

% 結果の表示
fprintf("LSTM: IAE=%.4f, ISE=%.4f, ITAE=%.4f, Tr=%.4f s, Ts=%.4f s, OS=%.2f%%, ESS=%.4f\n", ...
    metrics_lstm.IAE, metrics_lstm.ISE, metrics_lstm.ITAE, metrics_lstm.Tr, metrics_lstm.Ts, metrics_lstm.OS, metrics_lstm.ESS);
fprintf(" PID: IAE=%.4f, ISE=%.4f, ITAE=%.4f, Tr=%.4f s, Ts=%.4f s, OS=%.2f%%, ESS=%.4f\n", ...
    metrics_pid.IAE, metrics_pid.ISE, metrics_pid.ITAE, metrics_pid.Tr, metrics_pid.Ts, metrics_pid.OS, metrics_pid.ESS);

%% ローカル関数
function metrics = compute_metrics(t, ref, y)
%COMPUTE_METRICS Calculate classical control metrics for step responses.
%   metrics = COMPUTE_METRICS(t, ref, y) returns a struct with the fields
%   IAE, ISE, ITAE, Tr, Ts, OS, ESS based on the discrete time series.

    if numel(t) ~= numel(ref) || numel(t) ~= numel(y)
        error('compute_metrics:InputSizeMismatch', 't, ref, y must have the same length.');
    end

    if any(diff(t) <= 0)
        error('compute_metrics:TimeNotIncreasing', 't must be strictly increasing.');
    end

    dt = [0; diff(t)];
    e = ref - y;

    metrics = struct();
    metrics.IAE = sum(abs(e) .* dt);
    metrics.ISE = sum((e .^ 2) .* dt);
    metrics.ITAE = sum(t .* abs(e) .* dt);

    ref_ss = ref(end);
    y_ss = y(end);

    idx10 = find(y >= 0.1 * ref_ss, 1, 'first');
    idx90 = find(y >= 0.9 * ref_ss, 1, 'first');
    if isempty(idx10) || isempty(idx90)
        metrics.Tr = NaN;
    else
        metrics.Tr = t(idx90) - t(idx10);
    end

    tol = 0.02 * abs(ref_ss);
    within = abs(y - ref_ss) <= tol;
    idx_ts = find(arrayfun(@(k) all(within(k:end)), 1:numel(y)), 1, 'first');
    if isempty(idx_ts)
        metrics.Ts = NaN;
    else
        metrics.Ts = t(idx_ts);
    end

    if ref_ss == 0
        metrics.OS = NaN;
    else
        metrics.OS = max(0, (max(y) - ref_ss) / ref_ss * 100);
    end

    metrics.ESS = ref_ss - y_ss;
end
