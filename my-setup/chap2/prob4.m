clear; clc; close all;

% 1. Parameters
A = 1;              % Amplitude
F0 = 2000;          % Frequency (Hz)
phi = 0;            % Phase
dur = 1.5;          % Playback duration (seconds)

Fs1 = 3 * F0;       % Fs1 = 6000 Hz (> 2*F0: No aliasing)
Fs2 = 1.5 * F0;     % Fs2 = 3000 Hz (< 2*F0: Aliasing occurs)

% 2. Signals for plotting (first 5 periods)
t_plot = 5 / F0;

tc = 0:1/(50*F0):t_plot;
xc = A * cos(2*pi*F0*tc + phi);

t1 = 0:1/Fs1:t_plot;
x1_plot = A * cos(2*pi*F0*t1 + phi);

t2 = 0:1/Fs2:t_plot;
x2_plot = A * cos(2*pi*F0*t2 + phi);

figure;
plot(tc, xc, 'k--', 'LineWidth', 1.2); hold on;
stem(t1, x1_plot, 'b', 'filled', 'MarkerSize', 5);
stem(t2, x2_plot, 'r', 'filled', 'MarkerSize', 5);
grid on;
legend('x(t) Analog', 'x1[n] (Fs1 = 3*F0)', 'x2[n] (Fs2 = 1.5*F0)');
xlabel('Time (s)');
ylabel('Amplitude');
title('Sampling of Sine Wave: Fs1 vs Fs2');

% 3. Audio signals generation
t1_audio = 0:1/Fs1:dur;
x1 = A * cos(2*pi*F0*t1_audio + phi);

t2_audio = 0:1/Fs2:dur;
x2 = A * cos(2*pi*F0*t2_audio + phi);

% 4. Recovery check display
fprintf('--- SAMPLING RECOVERY ANALYSIS ---\n');
fprintf('Original F0 = %d Hz\n', F0);
fprintf('1. Fs1 = %d Hz (3*F0)   -> Fs1 > 2*F0: RECOVERABLE (No aliasing)\n', Fs1);
fprintf('2. Fs2 = %d Hz (1.5*F0) -> Fs2 < 2*F0: UNRECOVERABLE (Aliasing to %d Hz)\n', Fs2, abs(F0 - Fs2));

% 5. Playback
sound(x1, Fs1);
pause(dur + 0.5);

sound(x2, Fs2);
