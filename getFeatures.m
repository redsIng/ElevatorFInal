function x = getFeatures(s, a, gridx, gridv, M, N, A)
% a:azione A: numero di azioni
% M+1 celle sulla x e sulla y, ad ogni cella si ha una feature
nCells = (M + 1)^2;
% Dimensione del vettore delle feature da addestrare
d = A*N*nCells;
x = zeros(d, 1);

for ii = 1 : N
    xxx = gridx(ii, :);
    vvv = gridv(ii, :);

    %Identificazione univoca del punto nelle griglie
    ix = find(s(1) >= xxx(1:end-1) & s(1) <= xxx(2:end), 1, 'first');
    iv = find(s(2) >= vvv(1:end-1) & s(2) <= vvv(2:end), 1, 'first');
    ind = sub2ind([M + 1, M + 1, N, A], ix, iv, ii, a);
    x(ind) = 1;
end

