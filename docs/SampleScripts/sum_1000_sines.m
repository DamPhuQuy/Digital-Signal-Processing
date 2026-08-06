% input parameters
L = 8000; % half-length of output signal (in samples)
N = 100; % number of sines w/ different frequencies

Fs = 8000;

w_step = 2*pi/N; % freq. step
n = -L:L; % discrete-time axis of signal

signal = zeros(1,length(n)); % create an empty signal
for w = -pi:w_step:pi % (N+1) frequencies between -pi & pi
   % add a sine w/ current freq. into the sum signal
   signal = signal + cos(w*n);
end

% the final signal = average of the summed signal
output = signal/(N+1); 
clf
plot(n(1:1000), abs(output(1:1000)));
grid;

delta = [zeros(1,L), 1, zeros(1,L)]; % xung don vi
sound(output, Fs)







