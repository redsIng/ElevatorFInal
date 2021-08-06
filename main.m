% Assignment Final - Minimum Time Lift

% The reinforcement learning problem is getting an optimal policy for controlling
% an elevator. In particular, the actions are represented by the possible increases
% of the forces that allow the movement of the lift, which for simplicity has been
% considered to have a unitary mass, while the states are all the possible positions
% that the lift can assume in space. Given the high cardinality of the states and
% the continuous nature of the problem, we have chosen to solve this problem with the
% application of the SARSA_ET typical of the functional approximation.
clc
clear
close all

%% Init

% Initial Point

yStart = 0;

% Action List

action = [-1,0,1];
% Lower and UpperBound to y position

lby = -2;
uby = 8;
% Lower and UpperBound to velocity

lbv = -6;
ubv = 6;
% Grid Dimension
% Aumentando la dimensione delle griglie la generalizzazione peggiora
% poihcé le celle sono più piccole
% M = # celle sulla x e sulla v
M =8;
% Number of grid
N = 20;
% Number of episodes
numEpisodes = 1e3;

epsilon = 1e-1;
alpha =1.e-2;
gamma = 1;
% lambda = 0 ->Sarsa lambda=1->MC
lambda = 0.8;

% Number of Cells
nCells = (M+1)^2;
d = length(action)*N*nCells;
% Costruction of tiles
% To build tiles we need to bound y position between intial and final
% point. Due to the nature of the problem we have to consider bounded
% velocity to safety of the people in the elevator

[gridx, gridv] = build_tiles(lby, uby,lbv,ubv, M, N);

% Init Elevatr Enviroment
env = ElevatorConcrete;

%Plotting Enviroment
%env.plot;

% History of all episodes
episode=[];
%% TRAINING PHASE - IMPLEMENTING SARSA ET ALGORITHM (backward view)

% Inizializzazione dei pesi arbitraria
w = zeros(d,1);

for ii = 1:numEpisodes
    % Vettore delle tracce di elegibilità
    z = zeros(d,1);
    %Inizializzazione dello stato - Punto Iniziale
    s = [0,0];
    %Scelta dell'azione in maniera epsilon greedy massimizzando la q
    a = epsgreedy(s, w, epsilon, gridx, gridv, M, N, action);
    isTerminal = false;
    % Numero di passi dall'inizio dell'episodio
    steps = 0;
    if ii == numEpisodes
        episode(end+1,:) = [s,a,0,0];
    end
    while ~isTerminal
        
        steps = steps + 1;
        x = getFeatures(s,a,gridx,gridv,M,N,length(action));
        % Dinamioca dell'ascensore: ritorna stato successivo, reward e se
        % mi trovo in uno stato terminale
        [sp, r, isTerminal] = env.step(s,action(a),0);
        if isTerminal
            % Differenza temporale
            delta = r-w'*x;
            this.act = 0;
        else
            % Calcolo dell'azione successiva e del nuovo vettore di pesi
            ap = epsgreedy(sp, w, epsilon, gridx, gridv, M, N, action);
            xp = getFeatures(sp,ap,gridx,gridv,M,N,length(action));
            delta = r+gamma*w'*xp-w'*x;
        end
        % Aggiornamento tracce di elegibilità x-> gradiente attuale
        z = gamma*lambda*z+x;
        % Aggiornamento dei pesi delle feature
        w = w+alpha*delta*z;
        s = sp;
        a = ap;
        
        env.PlotValue = 1;
        if ii == numEpisodes
                episode(end+1,:) = [s,a,r,env.act];
                pause(0.1)
                env.PlotValue = 1;
                env.State = [s(1),s(2)];
                env.plot        
        end
    end
    disp([ii, steps])
end
save ElevatorData episode


