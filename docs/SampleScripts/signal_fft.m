%%% plot magnitude spectrum of signal using FFT
% generate signal
Fs=10000;  % in Hz
F1=1000;    % in Hz
F2=2000;    % in Hz
F3=3000;    % in Hz
duration=2; % in seconds
nSamples=100;% number of signal samples for plotting

t=0:1/Fs:duration; % sampling the signal w/ sampling frequency Fs & limiting the signal's length
y=3*sin(2*pi*F1*t)+2*sin(2*pi*F2*t)+1*sin(2*pi*F3*t); % signal consists of 3 frequencies F1, F2, F3
sound(y,Fs);

figure(1)

% normalize signal magnitude into a range of [-1,1]
max_value=max(abs(y));
y=y/max_value;

% create time base
t=1/Fs:1/Fs:(length(y)/Fs);
subplot(3,1,1);plot(t(1:nSamples),y(1:nSamples),'k','LineWidth',1);
title('Signal waveform');
xlabel('Time (seconds)');
ylabel('Normalized magnitude');

% plot linear magnitude spectrum of signal using FFT
N_FFT = 2048; % number of frequency samples = 2^m
magSpec = abs(fft(y, N_FFT)); % get N_FFT samples of magnitude spectrum
k=1:N_FFT; % k axis
w=k*Fs/N_FFT; % frequency axis
subplot(3,1,2);plot(w(1:N_FFT/2), magSpec(1:N_FFT/2),'k','LineWidth',1);
title('One-sided Linear Magnitude Spectrum');
xlabel('Frequency (Hz)')
ylabel('Magnitude');

% plot log magnitude spectrum of signal using FFT
logmagSpec = 20*log10(magSpec);
subplot(3,1,3);plot(w(1:N_FFT/2), logmagSpec(1:N_FFT/2),'k','LineWidth',1);
title('One-sided Linear Magnitude Spectrum');
xlabel('Frequency (Hz)')
ylabel('Magnitude (dB)');


