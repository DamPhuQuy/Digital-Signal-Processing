clear; clc; close all;

[x, fs] = audioread('a.wav');
N = length(x);

t = (0:N-1) / fs;
plot(t, x);
grid on;
xlabel('thoi gian (s)');
ylabel('bien do am thanh');

sound(x, fs);
pause;

sound(x, fs / 2);
pause;

sound(x, 2 * fs);
