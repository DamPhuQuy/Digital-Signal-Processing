% Calculate FFT and draw frequency-domain representation of a signal

% Create time base of signal x(n)
n=0:0.1:20; % discrete-time base of signal

% Create value sequence of signal x(n)
%signal=3*cos(n); % x(n) = 3cos(n)
%signal=3*cos(n)+2*cos(5*n); % x(n) = 3cos(n)+2cos(5n)
signal=3*cos(n)+2*cos(5*n)+cos(10*n); % x = 3cos(n)+2cos(5n)+cos(10n)

N=length(signal);
t=linspace(0,1,N);
figure(1)
subplot(2,1,1);plot(t,signal)
grid
title('time-domain representation of signal x(n)') 


% calculate the FFT
T_max=1;
Fs=N/T_max;
NFFT=2^nextpow2(N);
f1 =(Fs/2)*linspace(0,1,NFFT/2+1);
y=(abs(fft(signal,NFFT)/N));

%plot(f1,abs(y(1:N/2)),'r')
subplot(2,1,2);plot(f1,y(1:NFFT/2+1))
axis([0 max(f1) 0 1.5])
grid
title('frequency-domain representation of signal X(f)')
legend('FT[x(n)]=X(f)')






