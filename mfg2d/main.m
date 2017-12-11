function main
% function main
%
%   The 2d MFG example from [1, Section 5.3]
%
%   Requires the path to ppfem:
%       addpath('../../lib/')
%
%   [1] https://hal.archives-ouvertes.fr/hal-01301282v1

%   R. Andreev, May 2016
    
    %%%%%%%%%%%%%%%%
    % MFG PROBLEM SETUP
    %%%%%%%%%%%%%%%%
    
    % Time horizon
    T = 1;
    
    % Spatial domain
    ax = -2.0; bx = +2.0; % left/right bdry in x
    ay = -1/2; by = +1/2; % bottom/top bdry in y
    
    % Tensorized FEM spaces with boundary conditions
    % "/1" = homogeneous Neumann   BCs
    % "/0" = homogeneous Dirichlet BCs
    SPACE_X = 'P2 x B2/1 x B2/1';
    SPACE_A = 'D1 x D0   x D0  ';
    SPACE_U = 'D0 x P1/0 x P0  '; % 1st component of B
    SPACE_V = 'D0 x P0   x P1/0'; % 2nd component of B
    SPACE_Z = 'D1 x D1   x D1  '; % Prox ambient space
    
    % Diffusion coefficient
    nu = 1;
    
    % Augmented Lagrangian parameter
    r = 1;

    % Discretization levels
    Lt = 4; Lx = 5; Ly = 4;
    
    % Number of ALG2 iterations
    alg2maxit = 10;
    
    % Initial density
    rho_0_x = @(x) (abs(ax-x) <= 1);
    rho_0_y = @(y) ones(size(y));
    rho_0 = @(x,y) rho_0_x(x) .* rho_0_y(y);
    
    % Preconditioner
    precond = @precond_basic;
%     precond = @precond_mg;
    
    % See also the functions
    %   hami_prox
    % and
    %   GammaStar
    
    %%%%%%%%%%%%%%%%
    % HELPERS
    %%%%%%%%%%%%%%%%
    
    % Matrix vector multiply (with possible diagnostic info)
    function y = MV(A, x); y = A * x; end

    % Process cell arrays
    foreach = @(S, f) cellfun(f, S, 'UniformOutput', false);
    
    % [x, y] = expand({x, y});
    expand = @(C) C{:};

    % "Vectorize" a tensor
    Vec = @(X) X(:);

    %%%%%%%%%%%%%%%%
    % FEM SPACES
    %%%%%%%%%%%%%%%%
    
    try % to see if ppfem is working
        addpath('ppfem/lib/');
        ppfem_construct([0 1], 0, 'L');
    catch
        msg = [];
        msg = [msg, 'Failed to call the ppfem constructor. '];
        msg = [msg, 'It may be necessary to set addpath(''..../ppfem/lib/''). '];
        %msg = [msg, 'If you do not have ppfem, download it from: '];
        %msg = [msg, 'http://bitbucket.org/numpde/ppfem'];
        error(msg);
    end
    
    % 1d meshes
    mesht = linspace(0,  T,  2^Lt + 1);
    meshx = linspace(ax, bx, 2^Lx + 1);
    meshy = linspace(ay, by, 2^Ly + 1);
    
%     % Mesh refinement at a location x, from above (d = 1) or from below (d = -1)
%     mesh_ref_at = @(m,x,d) sort(unique([m, x + Vec(d(:) * min(nonzeros(abs(m-x))) * 2.^(-(0:10)))']));

    
    % Wrapper to construct FEM spaces incl mass matrix
    withmass = @(V) setfield(V, 'M', ppfem_gramian(V));
    
    % Convert a string like '0' to a boundary condition function handle
    function BC = get_BC(type)
        switch type
            case '0' % Homogeneous Dirichlet BCs
                BC = @(S) ppfem_homconstraint(S, ppfem_eval(S, S.x([1,end])));
            case '1' % Homogeneous Neumann BCs
                BC = @(S) ppfem_homconstraint(S, ppfem_eval(ppfem_transform(S, @fnder), S.x([1,end])));
            case '8' % Periodic C1
                BC_per_r = @(S,r) ppfem_homconstraint(S, diff(ppfem_eval(ppfem_transform(S, @(p)fnder(p,r)), S.x([1,end]))));
                BC = @(S) BC_per_r(BC_per_r(S,0),1);
            case '-' % No BCs
                BC = @(S) S;
            otherwise
                error(['Unknown boundary condition type: ' type]);
        end
    end
    
    % Parse e.g. type = 'P1/0' to produce the 1d FEM space
    function S = FEM_1(type, mesh)
        spec = strsplit(strtrim(type), '/'); % spec has length at least 1
        TP = spec{1}; TP = struct('ppfemtype', TP(1), 'degree', str2num(TP(2)));
        if (length(spec) > 1); bc = get_BC(spec{2}); else bc = @(x)x; end
        S = withmass(bc(ppfem_construct(mesh, TP.degree, TP.ppfemtype)));
    end
    
    % Parse e.g. types = 'P1 x B2/1 x B2/1' to construct the FEM spaces
    function varargout = FEM_N(types, meshes)
        varargout = cellfun(@FEM_1, strsplit(types, 'x'), meshes, 'UniformOutput', false);
    end

    % Collect the univariate meshes
    meshes = {mesht, meshx, meshy};
    
    % Cost phi, density rho, velocity field m = (u, v, ..)
    [s_phi_t, s_phi_x, s_phi_y] = FEM_N(SPACE_X, meshes);
    [s_rho_t, s_rho_x, s_rho_y] = FEM_N(SPACE_A, meshes);
    [s_m_u_t, s_m_u_x, s_m_u_y] = FEM_N(SPACE_U, meshes);
    [s_m_v_t, s_m_v_x, s_m_v_y] = FEM_N(SPACE_V, meshes);
    
    % Broken space (ideally) encompassing rho, m and phi
    [s_max_t, s_max_x, s_max_y] = FEM_N(SPACE_Z, meshes);
    
    % Test basic memory requirements
    XYZ = {zeros([s_phi_y.dim, s_phi_x.dim, s_phi_t.dim]), zeros([s_rho_y.dim, s_rho_x.dim, s_rho_t.dim]), zeros([s_max_y.dim, s_max_x.dim, s_max_t.dim])};
    [x, y, z] = expand(foreach(XYZ, @(X) num2str(numel(X))));
    disp(['dim of phi / rho / max space: ' x ' / ' y ' / ' z]);
    clear XYZ x y z;

    %%%%%%%%%%%%%%%%
    % FEM SPACE -- DERIVATES
    %%%%%%%%%%%%%%%%
    
    D = @(S) ppfem_transform(S, @fnder);
    % Derivatives
    s_phi_t_1 = withmass(D(s_phi_t));
    s_phi_x_1 = withmass(D(s_phi_x)); s_phi_x_2 = withmass(D(s_phi_x_1));
    s_phi_y_1 = withmass(D(s_phi_y)); s_phi_y_2 = withmass(D(s_phi_y_1));

    % Projection matrix A <-- B
    P = @(A, B) (A.M \ ppfem_gramian(A, B));
    
    % Projections 
    Pt1 = P(s_rho_t, s_phi_t);
    Px0 = P(s_rho_x, s_phi_x); Py0 = P(s_rho_y, s_phi_y); 
    Px2 = P(s_max_x, s_phi_x); Py2 = P(s_max_y, s_phi_y); 
    Ut0 = P(s_m_u_t, s_phi_t); Vt0 = P(s_m_v_t, s_phi_t);

    % Derivative with projection
    Dt1 = P(s_rho_t, s_phi_t_1); 
    Dx2 = P(s_rho_x, s_phi_x_2); Dy2 = P(s_rho_y, s_phi_y_2); 
    Dx1 = P(s_m_u_x, s_phi_x_1); Dy1 = P(s_m_v_y, s_phi_y_1); 
    
    % Norm of the V space
    normV_2 = 0;                                 % As matrix
    normV_2 = normV_2 + kron(s_phi_x.M, s_phi_y.M);
    normV_2 = normV_2 + (nu^2) * (kron(s_phi_x_1.M, s_phi_y.M) + kron(s_phi_x.M, s_phi_y_1.M));
    normV_2_apply = @(v) normV_2 * v;            % As Riesz mapping
    normV = @(v) sqrt(dot(normV_2_apply(v), v)); % As norm ||o||_V
    
    % Temporal evaluators at t=0 and t=T
    s_phi_t_E0 = ppfem_eval(s_phi_t, 0);
    s_phi_t_ET = ppfem_eval(s_phi_t, T);
    
    % Template for a zero phi function (space-time)
    zero_phi = zeros((s_phi_y.dim * s_phi_x.dim), s_phi_t.dim);
    
    % Plot a (1+1)d FEM function where e.g. spaces = { spacet, spacex }
    % The function handle plotter defaults to @surf
    function h = plot_XxY(u, spaces, plotter)
        if (nargin < 3); plotter = @(varargin) surf(varargin{:}, 'LineStyle', 'none'); end
        [spacex, spacey] = spaces{:};
        ex = linspace(spacex.x(1), spacex.x(end), (109+1)); Ex = ppfem_eval(spacex, ex);
        ey = linspace(spacey.x(1), spacey.x(end), (107+1)); Ey = ppfem_eval(spacey, ey);
        h = plotter(ex, ey, Ey * reshape(u, [spacey.dim, spacex.dim]) * Ex'); 
        colormap summer; colorbar;
    end

    % Plotting shorthands
    plot_phi = @(u)   plot_XxY(u', {s_phi_x, s_phi_y});
    plot_rho = @(u)   plot_XxY(u', {s_rho_x, s_rho_y});
    cont_rho = @(u,c) plot_XxY(u', {s_rho_x, s_rho_y}, @(x,y,v)contour(x,y,v,c));
    
    %%%%%%%%%%%%%%%%
    % The space-time operators
    %%%%%%%%%%%%%%%%
    
    % The scaled Laplacian (projected)
    nuLaplace = (nu^2) * (kron(Dx2, Py0) + kron(Px0, Dy2));
    
    % The Lambda operator
    function sigma = Lambda(phi)
        phi = reshape(phi, size(zero_phi)); % Check size consistency
        sigma.a = (kron(Px0, Py0) * phi * Dt1') + (nuLaplace * phi * Pt1');
        sigma.u = (kron(Dx1, Py0) * phi * Ut0'); 
        sigma.v = (kron(Px0, Dy1) * phi * Vt0'); 
        sigma.c = -phi * s_phi_t_ET';
    end

    % The Lambda transposed operator
    function phi = LambdaT(sigma)
        phi = zero_phi;
        phi = phi + (kron(Px0, Py0)' * sigma.a * Dt1) + (nuLaplace' * sigma.a * Pt1);
        phi = phi + (kron(Dx1, Py0)' * sigma.u * Ut0);
        phi = phi + (kron(Px0, Dy1)' * sigma.v * Vt0);
        phi = phi - (sigma.c * s_phi_t_ET);
    end

    % The Riesz operator Y --> Y'
    function sigma = MY(sigma)
        sigma.a = kron(s_rho_x.M, s_rho_y.M) * sigma.a * s_rho_t.M';
        sigma.u = kron(s_m_u_x.M, s_m_u_y.M) * sigma.u * s_m_u_t.M';
        sigma.v = kron(s_m_v_x.M, s_m_v_y.M) * sigma.v * s_m_v_t.M';
        sigma.c = normV_2_apply(sigma.c);
    end
    

    %%%%%%%%%%%%%%%%
    % Proximal maps
    %%%%%%%%%%%%%%%%

    function SN = spectral_nodes_1d(SPC)
        get_x = @(S) [S(:).('x')]; % Extracts the field 'x'
        % Use the quadrature rule of SPC to obtain spectral nodes
        SN.x = get_x(SPC.QR(SPC.x([1:end-1; 2:end])'));
        SN.E = ppfem_eval(SPC, SN.x);
        SN.M = SPC.M;
    end

    function SN = spectral_nodes_2d(SPXY)
        SN.snx = spectral_nodes_1d(SPXY{1}); 
        SN.sny = spectral_nodes_1d(SPXY{2}); 
        % Space-time spectral nodes. With ndgrid use the order ".., y, x, t"
        [SN.y, SN.x] = ndgrid(SN.sny.x, SN.snx.x);
        SN.E = kron(SN.snx.E, SN.sny.E);
        SN.M = kron(SN.snx.M, SN.sny.M);
    end

    spectral_xy = spectral_nodes_2d({s_max_x, s_max_y});
    
    % Gamma*
    %
    % Used to compute the integral of a piecewise polynomial function:
    load_h_flat = kron(ppfem_assemload(@(x)1, {s_max_x}), ppfem_assemload(@(x)1, {s_max_y}));
    %
    function [d0, d1, d2] = GammaStar_reg(v, h_bar)
        % v is the vector of coefficients
        % of a (spcx_B2 x spcy_B2) function
        function [d0, d1, d2] = GammaStar(c, x, y)
            N_gamma = 1e3;
            N_rho1 = (abs(bx-x) <= 1);
            %N_rho1 = 1 / (bx-ax); % Uniform desired state

            case1 = ((c + N_gamma * N_rho1) >= 0); 
            cases = @(a, b) (a .* case1) + (b .* (~case1));

            d0_case1 = (1/N_gamma) * ((1/2) * (c.^2))  +  (c .* N_rho1);
            d1_case1 = (1/N_gamma) * (         c    )  +  (     N_rho1);
            d2_case1 = (1/N_gamma) * ones(size(c));
            d0_case2 = -N_gamma * (1/2) * (N_rho1.^2) .* ones(size(c));
            d1_case2 = 0;
            d2_case2 = 0;

            d0 = cases(d0_case1, d0_case2);
            d1 = cases(d1_case1, d1_case2);
            d2 = cases(d2_case1, d2_case2);
        end
        
        w = spectral_xy.E' \ load_h_flat;   % Integration weights
        V = spectral_xy.E * kron(Px2, Py2); % Evaluates on spectral nodes

        [d0, d1, d2] = GammaStar(V * v, spectral_xy.x(:), spectral_xy.y(:));
        
        spdiag = @(v) spdiags(v(:), 0, numel(v), numel(v));

        d0 = dot(w, d0) + (r/2) * normV(h_bar - v)^2;
        d1 = (V' * (w .* d1)) + r * (normV_2 * (v - h_bar));
        d2 = (V' * spdiag(w .* d2) * V) + r * normV_2;
    end

    %

    % Note: SPTX = {spacet, spacex};
    function SN = spectral_nodes_txy(SPTXY)
        [spct, spcx, spcy] = SPTXY{:};                            % Small space
        [SPCT, SPCX, SPCY] = expand({s_max_t, s_max_x, s_max_y}); % Large space

        SN.snt = spectral_nodes_1d(SPCT); 
        SN.snx = spectral_nodes_1d(SPCX); 
        SN.sny = spectral_nodes_1d(SPCY);
        SN.EX = kron(SN.snx.E, SN.sny.E);
        SN.MX = kron(SN.snx.M, SN.sny.M);

        % Space-time spectral nodes. With ndgrid use the order ".., y, x, t"
        [SN.y, SN.x, SN.t] = ndgrid(SN.sny.x, SN.snx.x, SN.snt.x);
        
        SN.Qt = ppfem_gramian(SPCT, spct);
        SN.QX = kron(ppfem_gramian(SPCX, spcx), ppfem_gramian(SPCY, spcy));

        % Evaluate U on the spectral nodes
        SN.nodal = @(U) (SN.EX * (SN.MX \ (SN.QX * U * SN.Qt') / SPCT.M) * SN.snt.E');
        % Reconstruct U from its values V on the spectral nodes
        SN.coeff = @(V) (SN.QX \ (SN.MX * (SN.EX \ V / SN.snt.E') * SPCT.M) / SN.Qt');
    end

    spectral_a = spectral_nodes_txy({s_rho_t, s_rho_x, s_rho_y});
    spectral_u = spectral_nodes_txy({s_m_u_t, s_m_u_x, s_m_u_y});
    spectral_v = spectral_nodes_txy({s_m_v_t, s_m_v_x, s_m_v_y});

    % A*
    %
    % function [a, b] = hami_prox(r, a0, b0, t, x) ==> IS DEFINED BELOW
    

    %%%%%%%%%%%%%%%%
    % PREPARE ALG2
    %%%%%%%%%%%%%%%%
    
    
    % Starting values for the ALG2
    phi = zero_phi; sigma = Lambda(phi); lambda = sigma;

    % Assemble initial datum rho(0) -- use the phi space!
    % General assembly call
    %load_rho_0 = (ppfem_assemload(rho_0, {s_phi_x, s_phi_y}));
    % Tensorized data: faster
    load_rho_0 = kron(ppfem_assemload(rho_0_x, {s_phi_x}), ppfem_assemload(rho_0_y, {s_phi_y}));
    
    % How is phi_T determined; is the final value phi(T) given?
    if (false)
        % Yes, phi(T) is given, assemble phi_T here
        %   Variant 1. General assembly call:
        %load_phi_T = ppfem_assemload(@(x,y) 0, {s_phi_x, s_phi_y});
        %   Variant 2. Shortcut if phi(T) ~ rho(0):
        load_phi_T = 0 * load_rho_0;
        %
        %   In any case:
        phi_T = (s_phi_x.M) \ load_phi_T; clear load_phi_T;
    else
        % No, phi(T) is not given. A final cost Gamma*(phi(T)) is present
        phi_T = []; % (= marker for undetermined phi_T)
    end

    % Sums up all fields of a Matlab struct
    struct_sum = @(S) sum(structfun(@(x)full(x), S));
    % Linear algebra operations on structs. op_fieldwise is defined below
    ADD = @(a,b) op_fieldwise(@plus, a, b);
    MUL = @(a,b) op_fieldwise(@times, a, b);
    DOT = @(a,b) struct_sum(op_fieldwise(@(p,q) dot(p(:),q(:)), a, b));
    

    %%%%%%%%%%%%%%%%
    % Preconditioner: Basic version
    %%%%%%%%%%%%%%%%
    
    % Temporal transformation
    eigf = @(A,B) eig(full(A), full(B)); 
    [Vt, Jt] = eigf((s_phi_t_1.M + (s_phi_t_ET' * s_phi_t_ET)), s_phi_t.M);
    CC.OMEGA2 = diag(Jt); % with log2-rounding: 2 .^ (round(log2(diag(Jt))));
    CC.Vt = Vt; 
    clear eigf Vt Jt;
    %
    Mx = s_phi_x.M; Ax = s_phi_x_1.M; Bx = s_phi_x_2.M;
    My = s_phi_y.M; Ay = s_phi_y_1.M; By = s_phi_y_2.M;
    CC.A = 0;
    CC.A = CC.A + (kron(Bx,My) + 2*kron(Ax,Ay) + kron(Mx,By)) * (nu^4);
    CC.A = CC.A + (kron(Ax,My) + kron(Mx,Ay));
    CC.M = kron(Mx,My);
    clear Mx Ax Bx My Ay By;
    %
    function phi = precond_basic(phi)
        phi = phi * CC.Vt;
        for kk = 1 : size(phi,2)
            phi(:,kk) = (CC.A + CC.OMEGA2(kk) * CC.M) \ phi(:,kk);
        end
        phi = phi * CC.Vt';
    end

    %%%%%%%%%%%%%%%%
    % Multigrid preconditioner
    %%%%%%%%%%%%%%%%
    
    % The preconditioner
    MGOP = get_MGOP(); 
    function phi = precond_mg(phi)
        phi = phi * CC.Vt;
        mgmset = struct('nu1', 5, 'nu2', 5, 'gam', 2); 
        phi = MG(MGOP, phi, 0*phi, mgmset);
        phi = phi * CC.Vt';
    end
    % 
    % Spatial multigrid components. See function MG below
    function OP = get_MGOP(OP, sz)
    % Construct all components of the multigrid
        if ((nargin == 0) || isempty(OP))
        % Construct operators on the finest level
            OP = struct('A', CC.A, 'M', CC.M, 'op', []);
            sz = [s_phi_x.dim, s_phi_y.dim];
        end
        
        % The direct solver (for the coarse spatial grid)
        function u = direct_solve(f, A, M)
            u = 0*f;
            for kk = 1 : size(f,2)
                u(:,kk) = (A + CC.OMEGA2(kk) * M) \ f(:,kk);
            end
        end
        
        % Coarse grid scenario
        if (~all(sz >= 2)); 
            OP.op = []; 
            OP.solve = @(f) direct_solve(f, OP.A, OP.M);
            return; 
        end

        % Main forward operator
        spdiag = @(v) spdiags(v(:), 0, numel(v), numel(v));
        W = spdiag(CC.OMEGA2);
        OP.apply = @(X) ((OP.A * X) + (OP.M * X) * W);
        
        % Multigrid smoothers
        solveMx = @(X) (OP.M \ X);
        %[L,U,P] = ilu(OP.M, struct('type', 'ilutp', 'droptol', 0.1)); % OP.M ~ L*U*P'
        %solveMx = @(X) (P * (U \ (L \ X)));
        lammax0 = eigs(@(x) OP.M \ (OP.A * x), prod(sz), [], 1, 'LM', struct('isreal', true, 'issym', false));
        L = sparse(diag(1 ./ (lammax0 + CC.OMEGA2)));
        OP.smooth1 = @(v, g) (v - (solveMx(OP.apply(v) - g) * L));
        OP.smooth2 = OP.smooth1;
        
        % Prolongation matrix for 1d B2 with Neumann BC
        % n = number of fine dofs = dimension of the fine B2 space
        function P = get_P_B2(n)
            P = spdiags(ones(n,1) * [1 2 1], -1:1, n, n);
            P = sparse(conv2(full(P), [1 1]));
            P = P(:, 2:2:end);
            P(1, 1) = 4;
            P(end, end) = 4;
            P = P / 4;
        end

        Px = get_P_B2(sz(1)); 
        Py = get_P_B2(sz(2)); 
        OP.P = kron(Px, Py); % Prolongation
        OP.R = OP.P';        % Restriction

        % Coarse grid recursion
        OP.op = get_MGOP(struct('A', OP.R * OP.A * OP.P, 'M', OP.R * OP.M * OP.P), [size(Px,2), size(Py,2)]);
    end

    %%%%%%%%%%%%%%%%
    % ITERATE ALG2
    %%%%%%%%%%%%%%%%

    figure(1);

    ALG2NIT = [1 : alg2maxit];
    for alg2nit = ALG2NIT
        disp(['Started ALG2 iteration #' num2str(alg2nit) '.']);
        
        % Duration of one ALG2 iteration
        alg2iter = tic;
        
        %%%%%%%%%%%%%%
        % ALG2 STEP A: Solve space-time PDE for phi
        %%%%%%%%%%%%%%
        
        alg2stepa = tic;
        
        % Operator
        AA = @(x) LambdaT(MY(Lambda(x)));
        % Right hand side
        bb = LambdaT(MY(ADD(sigma, MUL(-1/r, lambda)))) - (1/r) * (load_rho_0 * s_phi_t_E0);

        % CG method with the previous phi as initial guess
        tol = 1e-1;
        maxit = 100;
        phi = phi + PCG(AA, (bb - AA(phi)), tol, maxit, precond, zero_phi);
        
        alg2stepa = toc(alg2stepa);
        disp(['Step A done in ' num2str(alg2stepa) 's']);
        
        %%%%%%%%%%%%%%
        % ALG2 STEP B: Prox operator to find sigma
        %%%%%%%%%%%%%%
        
        % Prior for the prox operator
        sigma_bar = ADD(Lambda(phi), MUL(1/r, lambda));
        
        % Substep 1/2 of STEP B: Prox for A*
        alg2stepb1 = tic;
        %
        % Retrieve the nodal values of the prior on the spectral nodes
        sigma_bar_a = spectral_a.nodal(sigma_bar.a); 
        sigma_bar_u = spectral_u.nodal(sigma_bar.u); 
        sigma_bar_v = spectral_v.nodal(sigma_bar.v);
        % Prox for A* on nodal values on the spectral nodes
        [A, B] = hami_prox(r, sigma_bar_a(:), [sigma_bar_u(:) sigma_bar_v(:)], spectral_a.t(:), [spectral_a.x(:), spectral_a.y(:)]);
        % Retrieve coefficients from the nodal values
        sigma.a = spectral_a.coeff(reshape(A(:,:), size(sigma_bar_a)));
        sigma.u = spectral_u.coeff(reshape(B(:,1), size(sigma_bar_u)));
        sigma.v = spectral_v.coeff(reshape(B(:,2), size(sigma_bar_v)));
        clear sigma_bar_a sigma_bar_u sigma_bar_v A B;
        %
        alg2stepb1 = toc(alg2stepb1);
        
        
        % Substep 2/2 of STEP B: Prox for Gamma*
        alg2stepb2 = tic;
        %
        if not(isempty(phi_T))
            % Here: phi(T) is given
            sigma.c = -phi_T; 
        else
            tol = 1e-10;              % Relative tolerance for the minimization
            h_bar = Vec(sigma_bar.c); % Prior
            h_pre = sigma.c;          % Prediction / initial guess
            options = struct('Algorithm', 'trust-region', 'GradObj', 'on', 'MaxFunEvals', 1e2, 'TolFun', 1e-6, 'Display', 'off', 'Hessian', 'on', 'HessMult', @MV);
            did_converge = false;
            for k = 1 : 10            % Multipass to determine h 
                %h = fminsearch(@(v)GammaStar_reg(v, h_bar), h_pre, options);
                h = fminunc(@(v)GammaStar_reg(v, h_bar), h_pre, options);
                did_converge = (normV(h - h_pre) <= tol * normV(h)); 
                if (did_converge); break; else h_pre = h; end
            end
            if (~did_converge); warning('Minimization tolerance not met.'); end
            %
            sigma.c = reshape(h, size(sigma.c));
        end
        % 
        alg2stepb2 = toc(alg2stepb2);
        
        disp(['Step B done in (' num2str(alg2stepb1) ' + ' num2str(alg2stepb2) ')s']);
        
        %%%%%%%%%%%%%%
        % ALG2 STEP C: Update the multiplier lambda
        %%%%%%%%%%%%%%
        
        alg2stepc = tic;
        
        congap = ADD(Lambda(phi), MUL(-1, sigma));
        lambda = ADD(lambda, MUL(r, congap));
        
        % Retrieve the density rho from the multiplier
        rho = lambda.a;
        
        alg2stepc = toc(alg2stepc);
        disp(['Step C done in ' num2str(alg2stepc) 's']);

        % ALG2 STEPS A-C completed.
        alg2iter = toc(alg2iter);
        disp(['Iteration #' num2str(alg2nit) ' of ALG2 done in ' num2str(alg2iter) 's']);
        
        %%%%%%%%%%%%%%
        % ALG2 STEP "D": Compute the slack
        %%%%%%%%%%%%%%
        
        gap = sqrt(DOT(congap, MY(congap)));
        disp(['Slack norm equals ' num2str(gap)]);
        
        %%%%%%%%%%%%%%
        % ALG2 STEP "E": PLOT 
        %%%%%%%%%%%%%%
        
        subplot(2,2, 1);
        plot_rho(rho * ppfem_eval(s_rho_t, 0)'); 
        xlabel('x'); ylabel('y'); title('rho(0)');
        
        subplot(2,2, 2);
        plot_rho(rho * ppfem_eval(s_rho_t, T)'); 
        xlabel('x'); ylabel('y'); title('rho(T)');

        subplot(2,2, 3);
        plot_phi(phi * ppfem_eval(s_phi_t, 0)'); 
        xlabel('x'); ylabel('y'); title('phi(0)');
        
        subplot(2,2, 4);
        plot_phi(phi * ppfem_eval(s_phi_t, T)'); 
        xlabel('x'); ylabel('y'); title('phi(T)');

        % Durchschnaufpause
        for ii = 1:3; disp(' '); end; pause(0.1);
    end

    % Show a "video" of the solution
    figure(2);
    pause(1);
    for t = [0 : 0.02 : T]
        s = num2str(t);
        
        subplot(2,1, 1);
        plot_rho(rho * ppfem_eval(s_rho_t, t, 'a')'); 
        xlabel('x'); ylabel('y'); title(['rho(' s ')']);
        colormap bone; view([0,90]); axis image;

        subplot(2,1, 2);
        plot_phi(phi * ppfem_eval(s_phi_t, t, 'a')'); 
        xlabel('x'); ylabel('y'); title(['phi(' s ')']);
        colormap bone; view([0,90]); axis image;
        
        pause(0.1);
    end
end

% The point Hamiltonian proximal map (vectorized implementation)
function [A, B] = hami_prox(r, A0, B0, t, X, Ai, Bi)
% Runs Newton's method to compute
% argmin { A*(a + H(b)) + (r/2) [ |a0 - a|^2 + |b0 - b|^2 ] }
%
% Ai, Bi is the initial guess
% x is of size [length(T), d]

    x = X(:,1); y = X(:,2); clear X;
    
    if (nargin < 6); Ai = A0; end % Default initial guess
    if (nargin < 7); Bi = B0; end % Default initial guess
    
    % The hamiltonian has the form
    % H(t,x,q) = Htx(t,x) + Hq(q)
    % with radially symmetric Hq(q) := Hq(|q|)
    %
    I = 0;
    % Tough terrain: 
    I = (abs(x) <= 1/2) .* (abs(y) <= 1/4);
    Htx = 0 - (1e3 * I); 
    %
    cc = 1/2; pp = 2;
    assert(pp >= 2);
    Hq_0 = @(q) cc * abs(q).^(pp);
    Hq_1 = @(q) cc * abs(q).^(pp-1) * pp .* sign(q);
    Hq_2 = @(q) cc * abs(q).^(pp-2) * pp * (pp-1);
    %
    A0 = A0 + Htx;
    Ai = Ai + Htx;

    N_gamma = 1; % Crowd-aversion factor
    N_rho1 = 0;

    % This is Z := A*
    % Here, A(rho) = (gamma/2) (rho - rho1)^2 if rho >= 0 and +oo else
    % (not to be confused with A = vector of a values)
    Z_case1 = @(z) (z >= -N_gamma * N_rho1);
    Z_case2 = @(z) (z <  -N_gamma * N_rho1);
    Z_0 = @(z) Z_case1(z) .* (z.^2 ./ (2*N_gamma) + z * N_rho1) + Z_case2(z) .* (-N_gamma * N_rho1^2 / 2);
    Z_1 = @(z) Z_case1(z) .* (z    ./ (  N_gamma) +     N_rho1);
    Z_2 = @(z) Z_case1(z) .* (1    ./ (  N_gamma)             );

    A = zeros(size(A0)); B = zeros(size(B0));
    N = length(A0);
    
    a0 = A0; b0 = sqrt(sum(B0.^2, 2));
    numit = 20;
    % Initial values for the Newton iteration
    a = Ai; b = sqrt(sum(Bi.^2, 2));
    for it = 1 : numit
        H0 = Hq_0(b); H1 = Hq_1(b); H2 = Hq_2(b);

        Z1 = Z_1(a + H0);
        Z2 = Z_2(a + H0);

        f1 = Z1       + r * (a - a0);
        f2 = Z1 .* H1 + r * (b - b0);
        
        df1 = Z2 + r;  
        df2 = Z2 .* H1;
        df4 = Z2 .* H1.^2 + Z1 .* H2 + r;

        idet = 1 ./ (df1 .* df4 - df2 .* df2);
        
        da = ((+df4) .* f1  +  (-df2) .* f2) .* idet;
        db = ((-df2) .* f1  +  (+df1) .* f2) .* idet;
        
        a = a - da;
        b = b - db;
    end
    
    A = a - Htx; B = 0*B0;
    I = (b0 > eps);
    spdiag = @(v) spdiags(v(:), 0, numel(v), numel(v));
    B(I,:) = spdiag(b(I) ./ b0(I)) * B0(I,:);
end

% Operate field-wise on Matlab structs
function S = op_fieldwise(OP, t, S)
    % If t and S are numeric
    if (isnumeric(S) && isnumeric(t)); S = OP(t, S); return; end
    % If S is numeric and t is not then swap them
    if (isnumeric(S) && isstruct(t)); S = op_fieldwise(OP, S, t); return; end
    assert(isstruct(S), 'S should be a struct here');
    FS = fieldnames(S); % S determines the resulting structure
    for nf = 1 : length(FS)
        f = FS{nf};
        if (isnumeric(t)); arg1 = t; else arg1 = t.(f); end
        if (isnumeric(S)); arg2 = S; else arg2 = S.(f); end
        S.(f) = OP(arg1, arg2);
    end
end

% Multigrid meta procedure. Ref: [Hackbusch, 2003, p.80]
function u = MG(OP, f, u, opt)
    if (nnz(f) == 0); u = 0*f; return; end             % Trivial rhs?
    if (isempty(OP.op)); u = OP.solve(f); return; end  % Coarsest grid
    for k = 1 : opt.nu1; u = OP.smooth1(u, f); end     % Presmoothing
    d = OP.R * (OP.apply(u) - f);                      % Residual
    v = 0*d;                                           % Recursion cycle:
    for k = 1 : opt.gam; v = MG(OP.op, d, v, opt); end % V if gam=1, W if gam=2
    u = u - (OP.P * v);                                % Correction
    for k = 1 : opt.nu2; u = OP.smooth2(u, f); end     % Postsmoothing
end
    
% % Preconditioned CG on tensors: Version 1
% function [x, flag, relres, iter] = PCG(A, b, tol, maxit, P, x)
%     % Mat() is 'inverse' to Vec()
%     Vec = @(X) X(:);
%     Mat = @(X) reshape(X, size(b));
%     A = @(x) Vec(A(Mat(x))); b = Vec(b);
%     P = @(x) Vec(P(Mat(x))); x = Vec(x);
%     [x, flag, relres, iter] = pcg(A, b, tol, maxit, P, [], x);
%     x = Mat(x);
% end

% Preconditioned CG on tensors: Version 2
function [x, niter, history] = PCG(A, b, tol, maxit, P, x)
% function [x, niter, history] = PCG(A, b, tol, maxit, P, x0)
%
%   CG implementation variant from
%   [Tobler 2012 (PhD thesis), p 77, Algorithm 9]
%
%   A = A(x) is a function handle
%
%   P = P(y) is a function handle to the left preconditioner
%   (may be supplied as the empty [])

    did_converge = false;
    history = [];

    % Preprocess missing arguments
    if (nargin < 5); P = []; end
    if (nargin < 6); x = []; end
    % Default values of the arguments
    if (isempty(P)); P = @(x)x; end
    if (isempty(x)); x = 0 * b; end

    DOT = @(a,b) dot(a(:), b(:));

    Ax = A(x); r = b - Ax; Pr = P(r); p = Pr;

    % Reference residual (for the stopping criterion)
    % This is not the initial residual DOT(P(r), r)
    rho1 = DOT(P(b), b);
        
    for niter = 1 : maxit
        % Residual & stopping criterion
        rho = DOT(Pr, r);
        if (sqrt(rho) <= tol * sqrt(rho1))
            did_converge = true; break;
        end
    
        Ap = A(p);
        x = x + p * ((DOT(b, p) - DOT(Ax, p)) / DOT(Ap, p));
        Ax = A(x); 
        r = b - Ax; 
        Pr = P(r);
        p = Pr - p * (DOT(Ap, Pr) / DOT(Ap, p));

%         cgstatus = [res1];
%         history{niter} = recorder(P2(x), cgstatus);
    end

    if (did_converge)
        disp(['PCG converged at iteration ' num2str(niter)]);
    else
        disp(['PCG failed to converge within ' num2str(niter) ' iterations']);
    end
end

