% L-shaped method for solving two-stage stochastic linear programs with recourse
%
%    minimize   c'*x + p1*(f1'*y1) + p2*(f2'*y2)
%    subject to A*x <= b
%               B1*x + D*y1 <= d1
%               B2*x + D*y2 <= d2
%               x >= e, y1 >= 0, y2 >= 0
%
% Source data: Birge J R, Louveaux F. Introduction to stochastic programming[M]. Springer Science & Business Media, 2011.
%              Part III Solution Methods, Section 5.1
%
%% Example 1
clear; clc; close all;

c = [100; 150];
p1 = 0.4; p2 = 0.6;
f1 = [-24; -28]; f2 = [-28; -32];
d1 = [0; 0; 500; 100]; d2 = [0; 0; 300; 300];
A = [1, 1]; b = 120;
B1 = [-60 0; 0 -80; 0 0; 0 0];
B2 = [-60 0; 0 -80; 0 0; 0 0];
D = [6 10; 8 5; 1 0; 0 1];
e = [40; 20];

% Solve this problem using YALMIP
x = sdpvar(2,1); y1 = sdpvar(2,1); y2 = sdpvar(2,1);
constr = [];
constr = [constr, A*x <= b];
constr = [constr, B1*x + D*y1 <= d1];
constr = [constr, B2*x + D*y2 <= d2];
constr = [constr, x >= e, y1 >=0, y2 >= 0];
obj = c'*x + p1*f1'*y1 + p2*f2'*y2;
opts = sdpsettings('solver', 'cplex', 'verbose', 0);
diag = optimize(constr, obj, opts);

if diag.problem ~= 0
    disp('The primal problem is infeasible or undounded');
end

% Initialization with feasible x
%
%    minimize   c'*x
%    subject to A*x <= b
%               x >= e
%
x = sdpvar(2,1);
constr = [];
constr = [constr, A*x <= b];
constr = [constr, x >= e];
obj = c'*x;
optimize(constr, obj, opts);
x = value(x);
t = -Inf;

lambda1 = [];
lambda2 = [];

while true
    % Solve subproblems
    %
    % primal:                                    dual:
    %    minimize   fi'*yi                         maximize   ui'*(di - Bi*x)
    %    subject to D*yi <= di - Bi*x              subject to D'*ui <= fi
    %               yi >= 0                                   ui <= 0
    %
    % dual
%     u1 = sdpvar(4,1);
%     constr1 = [];
%     constr1 = [constr1, D'*u1 <= f1];
%     constr1 = [constr1, u1 <= 0];
%     obj1 = u1'*(d1 - B1*x);
%     diag1 = optimize(constr1, -obj1, opts);
%     lambda1 = [lambda1, value(u1)];
%     
%     u2 = sdpvar(4,1);
%     constr2 = [];
%     constr2 = [constr2, D'*u2 <= f2];
%     constr2 = [constr2, u2 <= 0];
%     obj2 = u2'*(d2 - B2*x);
%     diag2 = optimize(constr2, -obj2, opts);
%     lambda2 = [lambda2, value(u2)];
    
    % primal
    y1 = sdpvar(2,1);
    constr1 = [];
    constr1 = [constr1, D*y1 <= d1 - B1*x];
    constr1 = [constr1, y1 >= 0];
    obj1 = f1'*y1;
    diag1 = optimize(constr1, obj1, opts);
    pi1 = -dual(constr1(1));
    lambda1 = [lambda1, pi1];
    
    y2 = sdpvar(2,1);
    constr2 = [];
    constr2 = [constr2, D*y2 <= d2 - B2*x];
    constr2 = [constr2, y2 >= 0];
    obj2 = f2'*y2;
    diag2 = optimize(constr2, obj2, opts);
    pi2 = -dual(constr2(1));
    lambda2 = [lambda2, pi2];
    
    % Check whether the optimal solution of subprolems is feasible in primal problem
%     w = p1*value(u1)'*(d1 - B1*x) + p2*value(u2)'*(d2 - B2*x);
    w = p1*pi1'*(d1 - B1*x) + p2*pi2'*(d2 - B2*x);
    if t >= w
        break;
    end
    
    % Solve master problem
    %
    %    minimize   c'*x + t
    %    subject to A*x <= b
    %               x >= e
    %               t >= p1*u1'*(d1 - B1*x) + p2*u2'*(d2 - B2*x): u1 in lambda1, u2 in lambda2
    %
    x = sdpvar(2,1); t = sdpvar(1,1);
    constr = [];
    constr = [constr, A*x <= b];
    constr = [constr, x >= e];
    if ( ~isempty(lambda1) && ~isempty(lambda2) )
        for i = 1:size(lambda1,2)
            constr = [constr, t >= p1*lambda1(:,i)'*(d1 - B1*x) + p2*lambda2(:,i)'*(d2 - B2*x)];
        end
    end
    obj = c'*x + t;
    optimize(constr, obj ,opts);
    
    % Update x and t
    x = value(x);
    t = value(t);
end

%% Example 2
%
%    minimize   p1*(f1'*y1) + p2*(f2'*y2) + p3*(f3'*y3)
%    subject to 0 <= x <= e
%               B*x + D*y1 = d1
%               B*x + D*y2 = d2
%               B*x + D*y3 = d3
%               y1 >= 0, y2 >= 0, y3 >= 0
%
clear; clc; close all;

p1 = 1/3; p2 = 1/3; p3 = 1/3;
f1 = [1; 1]; f2 = [1; 1]; f3 = [1; 1];
e = 10;
B = [1];
D = [1, -1];
d1 = [1]; d2 = [2]; d3 = [4];

% Solve this problem using YALMIP
x = sdpvar(1,1); y1 = sdpvar(2,1); y2 = sdpvar(2,1); y3 = sdpvar(2,1);
constr = [];
constr = [constr,0 <= x <= e];
constr = [constr, B*x + D*y1 == d1];
constr = [constr, B*x + D*y2 == d2];
constr = [constr, B*x + D*y3 == d3];
constr = [constr, y1 >= 0, y2 >= 0, y3 >= 0];
obj = p1*f1'*y1 + p2*f2'*y2 + p3*f3'*y3;
opts = sdpsettings('solver', 'cplex', 'verbose', 0);
diag = optimize(constr, obj, opts);

if diag.problem ~= 0
    disp('The primal problem is either infeasible or unbounded');
end

% Initialization with feasible x
x = zeros(1,1);
t = -Inf;

lambda1 = []; lambda2 = []; lambda3 = [];

while true
    % Solve subproblems
    %
    % primal:                                    dual:
    %    minimize   fi'*yi                          maximize   ui'*(di - B*x)
    %    subject to D*yi = di - B*x                 subject to D'*ui <= fi
    %               yi >= 0
    %
    % primal
    y1 = sdpvar(2,1);
    constr1 = [];
    constr1 = [constr1, D*y1 == d1 - B*x];
    constr1 = [constr1, y1 >= 0];
    obj1 = f1'*y1;
    optimize(constr1, obj1, opts);
    pi1 = -dual(constr1(1));
    lambda1 = [lambda1, pi1];
    
    y2 = sdpvar(2,1);
    constr2 = [];
    constr2 = [constr2, D*y2 == d2 - B*x];
    constr2 = [constr2, y2 >= 0];
    obj2 = f2'*y2;
    optimize(constr2, obj2, opts);
    pi2 = -dual(constr2(1));
    lambda2 = [lambda2, pi2];
    
    y3 = sdpvar(2,1);
    constr3 = [];
    constr3 = [constr3, D*y3 == d3 - B*x];
    constr3 = [constr3, y3 >= 0];
    obj3 = f3'*y3;
    optimize(constr3, obj3, opts);
    pi3 = -dual(constr3(1));
    lambda3 = [lambda3, pi3];
    
    % dual
%     u1 = sdpvar(1,1);
%     constr1 = [];
%     constr1 = [constr1, D'*u1 <= f1];
%     obj1 = u1'*(d1 - B*x);
%     optimize(constr1, -obj1, opts);
%     lambda1 = [lambda1, value(u1)];
%     
%     u2 = sdpvar(1,1);
%     constr2 = [];
%     constr2 = [constr2, D'*u2 <= f2];
%     obj2 = u2'*(d2 - B*x);
%     optimize(constr2, -obj2, opts);
%     lambda2 = [lambda2, value(u2)];
%     
%     u3 = sdpvar(1,1);
%     constr3 = [];
%     constr3 = [constr3, D'*u3 <= f3];
%     obj3 = u3'*(d3 - B*x);
%     optimize(constr3, -obj3, opts);
%     lambda3 = [lambda3, value(u3)];
    
    % Check whether the optimal solution of subprolems is feasible in primal problem
    w = p1*pi1'*(d1 - B*x) + p2*pi2'*(d2 - B*x) + p3*pi3'*(d3 - B*x);
%     w = p1*value(u1)'*(d1 - B*x) + p2*value(u2)'*(d2 - B*x) + p3*value(u3)'*(d3 - B*x);
    if t - w >= -1e-5
        break;
    end
    
    % Solve master problem
    %
    %    minimize   t
    %    subject to 0 <= x <= e
    %               t >= p1*u1'*(d1 - B*x) + p2*u2'*(d2 - B*x) + p3*u3'*(d3 - B*x)
    %
    x = sdpvar(1,1); t = sdpvar(1,1);
    constr = [];
    constr = [constr, 0<= x <= e];
    if ( ~isempty(lambda1) && ~isempty(lambda2) && ~isempty(lambda3) )
        for i = 1:size(lambda1,2)
            constr = [constr, t >= p1*lambda1(:,i)'*(d1 - B*x) + p2*lambda2(:,i)'*(d2 - B*x) + p3*lambda3(:,i)'*(d3 - B*x)];
        end
    end
    obj = t;
    optimize(constr, obj, opts);
    
    % Update x and t
    x = value(x);
    t = value(t);
end

%% Example 3
%
%    minimize   c'*x + p1*(f1'*y1) + p2*(f2'*y2)
%    subject to T*x + W*y1 <= h1
%               T*x + W*y2 <= h2
%               x >= 0, y1, y2 >= 0
%
clear; clc; close all;

c = [3; 2];
p1 = 1/2; p2 = 1/2;
f1 = [-15; -12]; f2 = [-15; -12];
T = [-1, 0; 0, -1; 0, 0; 0, 0; 0, 0; 0, 0];
W = [3, 2; 2, 5; -1, 0; 1, 0; 0, -1; 0, 1];
h1 = [0; 0; -4.8; 6; -6.4; 8];
h2 = [0; 0; -3.2; 4; -3.2; 4];

% Solve this problem using YALMIP
x = sdpvar(2,1);
y1 = sdpvar(2,1); y2 = sdpvar(2,1);
constr = [];
constr = [constr, T*x + W*y1 <= h1];
constr = [constr, T*x + W*y2 <= h2];
constr = [constr, x >=0, y1 >= 0, y2 >= 0];
obj = c'*x + p1*f1'*y1 + p2*f2'*y2;
opts = sdpsettings('solver', 'cplex', 'verbose', 0);
diag = optimize(constr, obj, opts);

if diag.problem ~= 0
    disp('The primal problem is either infeasible or unbounded');
end

% Initialization with feasible x
x = sdpvar(2,1);
t = -Inf;
constr = [];
constr = [constr, x >= 0];
obj = c'*x;
optimize(constr, obj, opts);
x = value(x);

fea1 = []; fea2 = [];
opt1 = []; opt2 = [];

e = ones(6,1);
I = eye(6,6);

while true
    % Solve feasibility subproblems
    %
    % primal:                                           dual:
    %    minimize   e'*v+ + e'*v-                          maximize   -ui'*(hi - T*x)
    %    subject to W*yi + I*v+ - I*v- <= hi - T*x :ui     subject to ui'*W >= 0
    %               yi >= 0, v+, v- >= 0                              0<= ui <= e
    %
    all_fea = true;
    while true
        % Subproblem 1
        y1 = sdpvar(2,1);
        v1 = sdpvar(6,1); v2 = sdpvar(6,1);
        constr = [];
        constr = [constr, W*y1 + I*v1 - I*v2 <= h1 - T*x];
        constr = [constr, y1 >= 0, v1 >= 0. v2 >= 0];
        obj1 = e'*v1 + e'*v2;
        optimize(constr, obj1, opts);
        
        if value(obj1) > 0
            all_fea = false;
            fea1 = [fea1, dual(constr(1))];
            break;
        end
        
        % Subproblem 2
        y2 = sdpvar(2,1);
        v1 = sdpvar(6,1); v2 = sdpvar(6,1);
        constr = [];
        constr = [constr, W*y2 + I*v1 - I*v2 <= h2 - T*x];
        constr = [constr, y2 >= 0, v1 >= 0, v2 >= 0];
        obj2 = e'*v1 + e'*v2;
        optimize(constr, obj2, opts);
        
        if value(obj2) > 0
            all_fea = false;
            fea2 = [fea2, dual(constr(1))];
            break;
        end
        
        % All subproblems are feasible
        if abs(value(obj1)) <= 1e-5 && abs(value(obj2)) <= 1e-5
            break;
        end
    end
    
    % Solve optimality subproblems
    %
    % primal:                                 dual:
    %    minimize   fi'*yi                      maximize   ui'*(hi - T*x)
    %    subject to W*yi <= hi - T*x :ui        subject to W'*ui <= fi
    %               yi >= 0                                ui <= 0
    %
    if all_fea
        % Subproblem 1
        y1 = sdpvar(2,1);
        constr = [];
        constr = [constr, W*y1 <= h1 - T*x];
        constr = [constr, y1 >= 0];
        obj = f1'*y1;
        optimize(constr, obj, opts);
        
        pi1 = -dual(constr(1));
        opt1 = [opt1, pi1];
        
        % Subproblem 2
        y2 = sdpvar(2,1);
        constr = [];
        constr = [constr, W*y2 <= h2 - T*x];
        constr = [constr, y2 >= 0];
        obj = f2'*y2;
        optimize(constr, obj, opts);
        
        pi2 = -dual(constr(1));
        opt2 = [opt2, pi2];
        
        % Check optimality
        % t >= p1*pi1'*(h1 - T*x) + p2*pi2'*(h2 - T*x)
        w = p1*pi2'*(h1 - T*x) + p2*pi2'*(h2 - T*x);
        if t >= w
            break;
        end
    end
    
    % Solve master problem
    %
    %    minimize   c'*x + t
    %    subject to x >= 0
    %               fea1'*(h1 - T*x) >= 0
    %               fea2'*(h2 - T*x) >= 0
    %               t >= p1*opt1'*(h1 - T*x) + p2*opt2'*(h2 - T*x)
    %
    x = sdpvar(2,1);
    t = sdpvar(1,1);
    constr = [];
    constr = [constr, x >= 0];
    
    % feasibility cut
    if ~isempty(fea1)
        for i = 1:size(fea1,2)
            constr = [constr, fea1(:,i)'*(h1 - T*x) >= 0];
        end
    end
    if ~isempty(fea2)
        for i = 1:size(fea2,2)
            constr = [constr, fea2(:,i)'*(h2 - T*x) >= 0];
        end
    end
    
    % optimality cut
    if ~isempty(opt1) && ~isempty(opt2)
        for i = 1:size(opt1,2)
            constr = [constr, t >= p1*opt1(:,i)'*(h1 - T*x) + p2*opt2(:,i)'*(h2 - T*x)];
        end
    end
    
    obj = c'*x + t;
    optimize(constr, obj, opts);
    
    % Update x and t
    x = value(x);
    t = value(t);
end