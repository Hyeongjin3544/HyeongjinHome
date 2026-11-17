function render_ols_fwl(whichClip)
% RENDER_OLS_FWL  Render 60fps 3b1b-inspired OLS & FWL projection videos.
% Outputs:
%   assets/media/ols-projection.mp4
%   assets/media/fwl-geometry.mp4

if nargin < 1 || isempty(whichClip)
    whichClip = 'all';
end

rootDir = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(rootDir, 'assets', 'media');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

if strcmpi(whichClip, 'all') || strcmpi(whichClip, 'ols')
    renderOlsClip(fullfile(outDir, 'ols-projection.mp4'));
end
if strcmpi(whichClip, 'all') || strcmpi(whichClip, 'fwl')
    renderFwlClip(fullfile(outDir, 'fwl-geometry.mp4'));
end
fprintf('All clips rendered successfully in %s\n', outDir);
end

function e = easeInOutCubic(t)
t = min(max(t, 0), 1);
if t < 0.5
    e = 4 * t^3;
else
    e = 1 - ((-2 * t + 2)^3) / 2;
end
end

function e = smootherstep(t)
t = min(max(t, 0), 1);
e = t^3 * (10 + t * (-15 + 6 * t));
end

function setOrbitCamera(ax, tgt, radius, azDeg, elDeg, camFovDeg)
az = deg2rad(azDeg);
el = deg2rad(elDeg);
pos = tgt + radius * [cos(el)*sin(az), -cos(el)*cos(az), sin(el)];
set(ax, 'CameraTargetMode', 'manual', 'CameraPositionMode', 'manual', ...
    'CameraUpVectorMode', 'manual', 'CameraViewAngleMode', 'manual', ...
    'CameraTarget', tgt, 'CameraPosition', pos, ...
    'CameraUpVector', [0 0 1], 'CameraViewAngle', camFovDeg, ...
    'Projection', 'perspective');
end

function c = blendCol(col, bgCol, alpha)
alpha = min(max(alpha, 0), 1);
c = bgCol + alpha * (col - bgCol);
end

function img = captureFrame(fig, targetH, targetW)
img = print(fig, '-RGBImage', '-r100');
if size(img, 1) ~= targetH || size(img, 2) ~= targetW
    ry = round(linspace(1, size(img, 1), targetH));
    rx = round(linspace(1, size(img, 2), targetW));
    img = img(ry, rx, :);
end
end

function drawVec3(ax, p0, p1, col, lw, headLen)
p0 = p0(:)';
p1 = p1(:)';
v = p1 - p0;
L = norm(v);
if L < 1e-4
    return;
end
plot3(ax, [p0(1) p1(1)], [p0(2) p1(2)], [p0(3) p1(3)], 'Color', col, 'LineWidth', lw);
u = v / L;
if abs(u(3)) < 0.9
    perp1 = cross(u, [0 0 1]);
else
    perp1 = cross(u, [1 0 0]);
end
perp1 = perp1 / norm(perp1);
perp2 = cross(u, perp1);
hl = min(headLen, 0.35 * L);
hw = 0.36 * hl;
base = p1 - hl * u;
pts = [p1; base + hw*perp1; base - hw*perp1; base + hw*perp2; base - hw*perp2];
faces = [1 2 4; 1 4 3; 1 3 5; 1 5 2];
patch(ax, 'Vertices', pts, 'Faces', faces, 'FaceColor', col, 'EdgeColor', 'none');
end

function drawRightAngle(ax, corner, dir1, dir2, sz, col)
corner = corner(:)';
dir1 = dir1(:)';
dir2 = dir2(:)';
d1 = dir1 / max(norm(dir1), 1e-8) * sz;
d2 = dir2 / max(norm(dir2), 1e-8) * sz;
pA = corner + d1;
pB = corner + d1 + d2;
pC = corner + d2;
plot3(ax, [pA(1) pB(1) pC(1)], [pA(2) pB(2) pC(2)], [pA(3) pB(3) pC(3)], ...
    'Color', col, 'LineWidth', 1.5);
end

function fig = makeBaseFig(bgCol)
fig = figure('Visible', 'off', 'Color', bgCol, 'Units', 'inches', ...
    'Position', [1 1 12.8 7.2], 'PaperUnits', 'inches', ...
    'PaperPosition', [0 0 12.8 7.2], 'PaperPositionMode', 'manual', ...
    'GraphicsSmoothing', 'on', 'InvertHardcopy', 'off');
end

function renderOlsClip(outPath)
bgCol   = [0.047 0.059 0.078]; % #0c0f14
gridCol = [0.22 0.25 0.32];
cX1     = [0.220 0.741 0.973]; % #38bdf8 cyan
cX2     = [0.176 0.831 0.749]; % #2dd4bf teal
cPlane  = [0.388 0.400 0.945]; % #6366f1 indigo
cY      = [0.984 0.749 0.141]; % #fbbf24 amber
cYhat   = [0.506 0.549 0.973]; % #818cf8 light indigo
cUhat   = [0.984 0.443 0.522]; % #fb7185 rose
cInk    = [0.96 0.96 0.98];
cMuted  = [0.64 0.66 0.72];

x1 = [2.2; 0.35; 0.15];
x2 = [0.55; 2.15; 0.25];
X  = [x1, x2];
bTrue = [0.95; 0.88];
uTrue = [-0.28; -0.35; 1.85];
% Force exact orthogonality to span(X)
Px = X * ((X' * X) \ X');
uTrue = uTrue - Px * uTrue;
uTrue = uTrue * (1.75 / norm(uTrue));
yhat = X * bTrue;
y = yhat + uTrue;
bHat = (X' * X) \ (X' * y);
comp1 = bHat(1) * x1;
comp2 = bHat(2) * x2;

fps = 60;
dur = 10.0;
nFrames = round(fps * dur);

vw = VideoWriter(outPath, 'MPEG-4');
vw.FrameRate = fps;
vw.Quality = 82;
open(vw);

fig = makeBaseFig(bgCol);
ax = axes(fig, 'Position', [0.05 0.06 0.90 0.76], 'Color', bgCol);

annotation(fig, 'textbox', [0.05 0.88 0.90 0.09], 'Tag', 'hudTitle', ...
    'String', 'OLS Projection Geometry: $y = P_X y + M_X y$  ($\hat{u}\perp\mathcal{C}(X)$)', ...
    'Interpreter', 'latex', 'Color', cInk, 'FontSize', 20, 'FontWeight', 'bold', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
hSub = annotation(fig, 'textbox', [0.05 0.82 0.90 0.06], 'Tag', 'hudSub', ...
    'String', '', 'Interpreter', 'latex', 'Color', cMuted, 'FontSize', 14, ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
lastSub = '';

for k = 1:nFrames
    t = (k - 1) / fps;
    cla(ax);
    hold(ax, 'on');
    axis(ax, 'off');
    xlim(ax, [-0.6 3.2]);
    ylim(ax, [-0.6 3.2]);
    zlim(ax, [-0.3 2.7]);
    daspect(ax, [1 1 1]);
    axis(ax, 'vis3d');

    % Smooth C2 360-deg orbit from 5.2s to 9.6s (locked camera prevents zoom wobble)
    azBase = 38;
    elBase = 23;
    if t >= 5.2 && t <= 9.6
        orb = smootherstep((t - 5.2) / 4.4);
        az = azBase + 360 * orb;
    else
        az = azBase;
    end
    setOrbitCamera(ax, [1.05 1.00 0.75], 14.5, az, elBase, 19);

    % Subtle base axes
    plot3(ax, [-0.3 2.9], [0 0], [0 0], 'Color', gridCol, 'LineWidth', 0.8);
    plot3(ax, [0 0], [-0.3 2.9], [0 0], 'Color', gridCol, 'LineWidth', 0.8);
    plot3(ax, [0 0], [0 0], [-0.2 2.4], 'Color', gridCol, 'LineWidth', 0.8);
    scatter3(ax, 0, 0, 0, 26, cInk, 'filled');

    % Phase 1: basis vectors & span plane
    pPlane = easeInOutCubic(t / 1.8);
    if pPlane > 0.01
        alphaP = 0.16 * pPlane;
        [s, r] = meshgrid(linspace(-0.25, 1.25, 2), linspace(-0.25, 1.25, 2));
        PX = s * x1(1) + r * x2(1);
        PY = s * x1(2) + r * x2(2);
        PZ = s * x1(3) + r * x2(3);
        surf(ax, PX, PY, PZ, 'FaceColor', cPlane, 'FaceAlpha', alphaP, ...
            'EdgeColor', cPlane, 'EdgeAlpha', 0.35 * pPlane, 'LineWidth', 0.8);
        % Span gridlines
        gridPlaneCol = blendCol(cPlane, bgCol, 0.22 * pPlane);
        for g = [0.0 0.5 1.0]
            pA = -0.2*x1 + g*x2; pB = 1.2*x1 + g*x2;
            plot3(ax, [pA(1) pB(1)], [pA(2) pB(2)], [pA(3) pB(3)], 'Color', gridPlaneCol, 'LineWidth', 0.6);
            pC = g*x1 - 0.2*x2; pD = g*x1 + 1.2*x2;
            plot3(ax, [pC(1) pD(1)], [pC(2) pD(2)], [pC(3) pD(3)], 'Color', gridPlaneCol, 'LineWidth', 0.6);
        end
        pX1 = pPlane * x1;
        pX2 = pPlane * x2;
        drawVec3(ax, [0;0;0], pX1, cX1, 2.5, 0.15);
        drawVec3(ax, [0;0;0], pX2, cX2, 2.5, 0.15);
        if pPlane > 0.7
            text(ax, x1(1)+0.08, x1(2)-0.12, x1(3), '$x_1$', 'Interpreter', 'latex', 'Color', cX1, 'FontSize', 16, 'FontWeight', 'bold');
            text(ax, x2(1)-0.12, x2(2)+0.10, x2(3), '$x_2$', 'Interpreter', 'latex', 'Color', cX2, 'FontSize', 16, 'FontWeight', 'bold');
            cLab = 1.15*x1 + 0.15*x2;
            text(ax, cLab(1), cLab(2), cLab(3)-0.08, '$\mathcal{C}(X)=\mathrm{span}(x_1,x_2)$', ...
                'Interpreter', 'latex', 'Color', cYhat, 'FontSize', 13);
        end
    end

    % Phase 2: target vector y
    pY = easeInOutCubic((t - 1.6) / 1.4);
    if pY > 0.01
        curY = pY * y;
        drawVec3(ax, [0;0;0], curY, cY, 3.0, 0.17);
        if pY > 0.75
            text(ax, y(1)+0.05, y(2)+0.05, y(3)+0.14, '$y$', 'Interpreter', 'latex', 'Color', cY, 'FontSize', 18, 'FontWeight', 'bold');
        end
    end

    % Phase 3: orthogonal projection drop y -> yhat + uhat
    pProj = easeInOutCubic((t - 3.2) / 2.0);
    if pProj > 0.01
        % Projection ray from y down to yhat
        rayEnd = y + pProj * (yhat - y);
        plot3(ax, [y(1) rayEnd(1)], [y(2) rayEnd(2)], [y(3) rayEnd(3)], ...
            '--', 'Color', blendCol(cUhat, bgCol, 0.90), 'LineWidth', 1.8);
        % Growing yhat vector in plane
        drawVec3(ax, [0;0;0], pProj * yhat, cYhat, 3.0, 0.17);
        if pProj > 0.85
            drawVec3(ax, yhat, y, cUhat, 2.6, 0.15);
            drawRightAngle(ax, yhat, -yhat, uTrue, 0.22, cInk);
            % Component decomposition parallelogram in col(X)
            plot3(ax, [comp1(1) yhat(1)], [comp1(2) yhat(2)], [comp1(3) yhat(3)], ':', 'Color', blendCol(cX2, bgCol, 0.70), 'LineWidth', 1.3);
            plot3(ax, [comp2(1) yhat(1)], [comp2(2) yhat(2)], [comp2(3) yhat(3)], ':', 'Color', blendCol(cX1, bgCol, 0.70), 'LineWidth', 1.3);
            text(ax, yhat(1)+0.10, yhat(2)+0.08, yhat(3)-0.12, '$\hat{y}=P_X y=X\hat{\beta}$', ...
                'Interpreter', 'latex', 'Color', cYhat, 'FontSize', 15, 'FontWeight', 'bold');
            midU = yhat + 0.52 * uTrue;
            text(ax, midU(1)+0.12, midU(2)+0.08, midU(3), '$\hat{u}=M_X y \perp \mathcal{C}(X)$', ...
                'Interpreter', 'latex', 'Color', cUhat, 'FontSize', 15, 'FontWeight', 'bold');
        end
    end

    if t < 3.2
        subStr = 'Step 1: Observation vector $y \in \mathbf{R}^n$ and column space $\mathcal{C}(X)=\mathrm{span}(x_1, x_2)$';
    elseif t < 5.2
        subStr = 'Step 2: Minimize $\|y - X\beta\|_2^2 \Leftrightarrow$ Orthogonal drop onto $\mathcal{C}(X)$ ($X''\hat{u}=0$)';
    else
        subStr = 'Pythagorean Decomposition: $\|y\|_2^2 = \|P_X y\|_2^2 + \|M_X y\|_2^2 \;\;(\mathrm{TSS}=\mathrm{ESS}+\mathrm{RSS})$';
    end
    if ~strcmp(subStr, lastSub)
        set(hSub, 'String', subStr);
        lastSub = subStr;
    end

    writeVideo(vw, captureFrame(fig, 720, 1280));
end

close(vw);
close(fig);
fprintf('Rendered %s\n', outPath);
end

function renderFwlClip(outPath)
bgCol   = [0.047 0.059 0.078]; % #0c0f14
gridCol = [0.22 0.25 0.32];
cX1     = [0.220 0.741 0.973]; % #38bdf8 cyan (control X1)
cX2     = [0.176 0.831 0.749]; % #2dd4bf teal (target X2)
cPlane  = [0.078 0.722 0.651]; % #14b8a6 teal plane C(X1)^perp
cY      = [0.984 0.749 0.141]; % #fbbf24 amber
cResX2  = [0.176 0.831 0.749];
cResY   = [0.984 0.749 0.141];
cProj2  = [0.506 0.549 0.973]; % #818cf8 indigo
cUhat   = [0.984 0.443 0.522]; % #fb7185 rose
cInk    = [0.96 0.96 0.98];
cMuted  = [0.64 0.66 0.72];

% X1 along vertical direction so C(X1)^perp plane reads cleanly in 3D
x1 = [0.0; 0.0; 2.15];
% Correlated x2 and y (strong component along x1)
x2 = [1.65; 0.35; 1.35];
M1 = eye(3) - (x1 * x1') / (x1' * x1);
x2Tilde = M1 * x2;
P1x2 = x2 - x2Tilde;

b1 = 0.65;
b2 = 1.38;
uTrue = [-0.28; 1.35; 0.0]; % Already in C(X1)^perp and near-orthogonal
X = [x1, x2];
uTrue = uTrue - X * ((X' * X) \ (X' * uTrue));
uTrue = uTrue * (1.28 / norm(uTrue));

y = b1 * x1 + b2 * x2 + uTrue;
yTilde = M1 * y;
P1y = y - yTilde;
b2Hat = (x2Tilde' * yTilde) / (x2Tilde' * x2Tilde);
yhatFwl = b2Hat * x2Tilde; % equals P_{M1 X2} M1 y

fps = 60;
dur = 12.0;
nFrames = round(fps * dur);

vw = VideoWriter(outPath, 'MPEG-4');
vw.FrameRate = fps;
vw.Quality = 82;
open(vw);

fig = makeBaseFig(bgCol);
ax = axes(fig, 'Position', [0.05 0.06 0.90 0.76], 'Color', bgCol);

annotation(fig, 'textbox', [0.05 0.88 0.90 0.09], 'Tag', 'hudTitle', ...
    'String', 'Frisch--Waugh--Lovell Theorem: Residual-on-Residual Projection', ...
    'Interpreter', 'latex', 'Color', cInk, 'FontSize', 20, 'FontWeight', 'bold', ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
hSub = annotation(fig, 'textbox', [0.05 0.82 0.90 0.06], 'Tag', 'hudSub', ...
    'String', '', 'Interpreter', 'latex', 'Color', cMuted, 'FontSize', 14, ...
    'EdgeColor', 'none', 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
lastSub = '';

for k = 1:nFrames
    t = (k - 1) / fps;
    cla(ax);
    hold(ax, 'on');
    axis(ax, 'off');
    xlim(ax, [-0.5 3.1]);
    ylim(ax, [-0.6 2.5]);
    zlim(ax, [-0.2 3.3]);
    daspect(ax, [1 1 1]);
    axis(ax, 'vis3d');

    % Smooth C2 360-deg orbit from 7.8s to 11.6s (locked camera prevents zoom wobble)
    azBase = 42;
    elBase = 24;
    if t >= 7.8 && t <= 11.6
        orb = smootherstep((t - 7.8) / 3.8);
        az = azBase + 360 * orb;
    else
        az = azBase;
    end
    setOrbitCamera(ax, [1.05 0.85 0.95], 14.5, az, elBase, 20);

    % Base orthogonal subspace plane C(X1)^perp (z = 0)
    [px, py] = meshgrid(linspace(-0.3, 2.7, 2), linspace(-0.3, 2.2, 2));
    surf(ax, px, py, zeros(size(px)), 'FaceColor', cPlane, 'FaceAlpha', 0.14, ...
        'EdgeColor', cPlane, 'EdgeAlpha', 0.35, 'LineWidth', 0.8);
    gridFwlCol = blendCol(cPlane, bgCol, 0.20);
    for g = [0.0 0.7 1.4 2.1]
        plot3(ax, [-0.2 2.6], [g g], [0 0], 'Color', gridFwlCol, 'LineWidth', 0.6);
    end
    for g = [0.0 0.9 1.8 2.6]
        plot3(ax, [g g], [-0.2 2.1], [0 0], 'Color', gridFwlCol, 'LineWidth', 0.6);
    end
    scatter3(ax, 0, 0, 0, 26, cInk, 'filled');
    text(ax, 0.10, -0.22, -0.06, '$\mathcal{C}(X_1)^\perp$', 'Interpreter', 'latex', 'Color', cPlane, 'FontSize', 15);

    % Phase 1: Show X1, X2, y
    pIntro = easeInOutCubic(t / 1.8);
    if pIntro > 0.01
        drawVec3(ax, [0;0;0], pIntro * x1, cX1, 2.6, 0.15);
        drawVec3(ax, [0;0;0], pIntro * x2, cX2, 2.3, 0.14);
        drawVec3(ax, [0;0;0], pIntro * y,  cY,  2.6, 0.15);
        if pIntro > 0.75
            text(ax, x1(1)-0.16, x1(2)-0.08, x1(3)+0.14, '$X_1$', 'Interpreter', 'latex', 'Color', cX1, 'FontSize', 16, 'FontWeight', 'bold');
            text(ax, x2(1)+0.06, x2(2), x2(3)+0.10, '$X_2$', 'Interpreter', 'latex', 'Color', cX2, 'FontSize', 15);
            text(ax, y(1)+0.06,  y(2),  y(3)+0.12,  '$y$',   'Interpreter', 'latex', 'Color', cY,  'FontSize', 16, 'FontWeight', 'bold');
        end
    end

    % Phase 2: Partial out X1 -> project onto C(X1)^perp
    pPart = easeInOutCubic((t - 2.2) / 2.6);
    if pPart > 0.01
        curX2 = x2 + pPart * (x2Tilde - x2);
        curY  = y  + pPart * (yTilde  - y);
        plot3(ax, [x2(1) curX2(1)], [x2(2) curX2(2)], [x2(3) curX2(3)], '--', 'Color', blendCol(cX2, bgCol, 0.75), 'LineWidth', 1.4);
        plot3(ax, [y(1)  curY(1)],  [y(2)  curY(2)],  [y(3)  curY(3)],  '--', 'Color', blendCol(cY,  bgCol, 0.75), 'LineWidth', 1.4);
        drawVec3(ax, [0;0;0], curX2, cResX2, 2.8, 0.15);
        drawVec3(ax, [0;0;0], curY,  cResY,  2.8, 0.15);
        if pPart > 0.85
            drawRightAngle(ax, x2Tilde, -x2Tilde, x1, 0.16, cMuted);
            drawRightAngle(ax, yTilde,  -yTilde,  x1, 0.16, cMuted);
            midX2T = 0.52 * x2Tilde;
            text(ax, midX2T(1)+0.06, midX2T(2)-0.28, -0.08, '$\tilde{X}_2=M_1 X_2$', ...
                'Interpreter', 'latex', 'Color', cResX2, 'FontSize', 15, 'FontWeight', 'bold');
            text(ax, yTilde(1)+0.16,  yTilde(2)+0.14,  0.12, '$\tilde{y}=M_1 y$', ...
                'Interpreter', 'latex', 'Color', cResY,  'FontSize', 15, 'FontWeight', 'bold');
        end
    end

    % Phase 3: Residual-on-residual projection inside C(X1)^perp
    pFwl = easeInOutCubic((t - 5.3) / 2.3);
    if pFwl > 0.01
        rayEnd = yTilde + pFwl * (yhatFwl - yTilde);
        plot3(ax, [yTilde(1) rayEnd(1)], [yTilde(2) rayEnd(2)], [yTilde(3) rayEnd(3)], ...
            '--', 'Color', blendCol(cUhat, bgCol, 0.90), 'LineWidth', 1.8);
        drawVec3(ax, [0;0;0], pFwl * yhatFwl, cProj2, 3.2, 0.16);
        if pFwl > 0.85
            drawVec3(ax, yhatFwl, yTilde, cUhat, 2.7, 0.15);
            drawRightAngle(ax, yhatFwl, -yhatFwl, yTilde - yhatFwl, 0.18, cInk);
            text(ax, yhatFwl(1)+0.10, yhatFwl(2)-0.22, 0.04, '$\hat{\beta}_2\tilde{X}_2$', ...
                'Interpreter', 'latex', 'Color', cProj2, 'FontSize', 15, 'FontWeight', 'bold');
            midU = 0.5*(yhatFwl + yTilde);
            text(ax, midU(1)+0.18, midU(2)-0.16, -0.08, '$\hat{u}=M_X y$', ...
                'Interpreter', 'latex', 'Color', cUhat, 'FontSize', 15, 'FontWeight', 'bold');
        end
    end

    if t < 2.2
        subStr = 'Full Regression: $y = X_1\hat{\beta}_1 + X_2\hat{\beta}_2 + \hat{u}$ (correlated $X_1, X_2$)';
    elseif t < 5.3
        subStr = 'Stage 1 (Partial Out $X_1$): $\tilde{X}_2 = M_1 X_2,\;\; \tilde{y} = M_1 y \;\in \mathcal{C}(X_1)^\perp$';
    else
        subStr = 'Stage 2: Regress $\tilde{y}$ on $\tilde{X}_2 \;\Rightarrow\; \hat{\beta}_2^{\mathrm{FWL}} = (\tilde{X}_2''\tilde{X}_2)^{-1}\tilde{X}_2''\tilde{y} \equiv \hat{\beta}_2^{\mathrm{Full}}$';
    end
    if ~strcmp(subStr, lastSub)
        set(hSub, 'String', subStr);
        lastSub = subStr;
    end

    writeVideo(vw, captureFrame(fig, 720, 1280));
end

close(vw);
close(fig);
fprintf('Rendered %s\n', outPath);
end
