% Create a test example of MILP
%
%    minimize   c'x + f'y
%    subject to Ax + By >= b
%               x >= 0
%               y in Y
%               y integer
%
% Source data: https://bookdown.org/edxu96/matrixoptim/benders-for-standard-milp.html
% 
clear; clc; close all;

c = [2; 6];
f = [2; 3];
A = [-1 2; 1 -3];
B = [3 -1; 2 2];
b = [5; 4];
M = -1000;

% Solve this problem using YALMIP
x = sdpvar(2,1); y = intvar(2,1);
constr = [];
constr = [constr, A*x + B*y >= b];
constr = [constr, x >= 0];
constr = [constr, 0<= y <= 2];
obj = c'*x + f'*y;
opts = sdpsettings('solver', 'cplex', 'verbose', 0);
diag = optimize(constr, obj, opts);

if diag.problem ~= 0
    disp('The primal problem is infeasible or unbounded');
end

% Initialization with feasible y
%    minimize   f'y + t
%    subject to y in Y
%               y integer
%               t >= M
%
LB = -Inf;
UB = +Inf;
eps = 1e-5;

y = intvar(2,1); t = sdpvar(1,1);
constr = [];
constr = [constr, 0 <= y <= 2];
constr = [constr, t >= M];
obj = f'*y + t;
optimize(constr, obj, opts);
y = value(y);

ray = [];
lambda = [];

while UB - LB >= eps
    % Solve subproblem
    % primal:                              dual:
    %    minimize   c'x                      maximize   (b - By)'u
    %    subject to Ax >= b - By             subject to A'u <= c
    %               x >= 0                              u >= 0
    %
    % dual
%     u = sdpvar(2,1);
%     constr = [];
%     constr = [constr, A'*u <= c];
%     constr = [constr, u >= 0];
%     obj = (b - B*y)'*u;
%     diag = optimize(constr, -obj, opts);
    
    % primal
    x = sdpvar(2,1);
    constr = [];
    constr = [constr, A*x >= b - B*y];
    constr = [constr, x >= 0];
    obj = c'*x;
    diag = optimize(constr, obj, opts);
    
    % The subproblem is either unbounded or feasible
    % unbounded ray:                      variant of subproblem:  < ----------------------- > dual problem:
    %    maximize 0                          minimize   ev+ + ev-                               maximize   (b - By)'*u
    %    subject to (b - By)'u == 1          subject to Ax + Iv+ - Iv- >= b - By: u             subject to A'u <= 0
    %               A'u <= 0                            x >= 0                                             u >= 0
    %               u >= 0                              v+ >= 0, v- >= 0
    %
    if diag.problem ~= 0
        % unbounded ray
%         u = sdpvar(2,1);
%         constr = [];
%         constr = [constr, (b - B*y)'*u == 1];
%         constr = [constr, A'*u <= 0];
%         constr = [constr, u >= 0];
%         obj = 0;
%         optimize(constr, -obj, opts);
%         ray = [ray, value(u)];
        
        % variant of subproblem
        x = sdpvar(2,1); v1 = sdpvar(2,1); v2 = sdpvar(2,1);
        constr = [];
        constr = [constr, A*x + v1 - v2 >= b - B*y];
        constr = [constr, x >=0, v1 >= 0, v2 >= 0];
        obj = ones(1,2)*v1 + ones(1,2)*v2;
        optimize(constr, obj, opts);
        ray = [ray, dual(constr(1))];
    else
%         lambda = [lambda, value(u)];
        lambda = [lambda, dual(constr(1))];
        UB = min(UB, f'*y + value(obj));
    end
    
    % Solve master problem
    %    minimize   f'y + t
    %    subject to y in Y
    %               y integer
    %               t >= M
    %               0 >= u'(b - By): u in ray
    %               t >= u'(b - By): u in lambda
    %
    y = intvar(2,1); t = sdpvar(1,1);
    constr = [];
    constr = [constr, 0<= y <= 2];
    constr = [constr, t >= M];
    if ~isempty(ray)
        for i = 1:size(ray,2)
            constr = [constr, ray(:,i)'*(b - B*y) <= 0];
        end
    end
    if ~isempty(lambda)
        for i = 1:size(lambda,2)
            constr = [constr, lambda(:,i)'*(b - B*y) <= t];
        end
    end
    obj = f'*y + t;
    optimize(constr, obj, opts);
    
    % Update lower bound and y
    LB = value(obj);
    y = value(y);
end