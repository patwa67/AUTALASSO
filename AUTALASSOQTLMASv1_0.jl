# ADMM adaptive lasso using ProximalOperators with line and golden section search
# Automatic tuning of learning rate and regularization parameter

using ProximalOperators
using DelimitedFiles
using Statistics
using LinearAlgebra

# Function for Proximal ADMM lasso with line search
function lasso_admm(Xtrainhot, ytrain, lam, theta, beta, f, abscovinv,tol; maxit=5000)
  u = zero(theta)
  grad = zero(theta)
  lamw = lam*abscovinv # Regularization parameter times weights
  g = NormL1(lamw) # Regularization function
  c = 0.5
  lr = 1.0
  loss(theta) = 0.5*norm(Xtrainhot*theta-ytrain)^2 # Loss function for line search
  for it = 1:maxit
    # Line search
    it % 8 == 1 && (grad = Xtrainhot'*(Xtrainhot*beta-ytrain))
    while  it % 20 == 2 && loss(theta) > (loss(beta) + grad'*(-beta) + (1.0/(2.0*lr))*norm(-beta)^2)
      lr = lr * c
      #println(lr)
    end
    gam = lr
    # ADMM perform f-update step
    prox!(beta, f, theta - u, gam)
    # ADMM perform g-update step
    prox!(theta, g, beta + u, gam)
    # Stopping criterion for ADMMM
    if norm(beta-theta, Inf) <= tol*(1+norm(u, Inf))
      break
    end
    # Dual update
    u .+= beta - theta
    #print(it)
  end
  return theta,beta,tol
end

# Function for Golden section search to optimize lambda
function gss_opt(alam, blam, tolgss, Xtesthot, ytest,abscovinv,maxnorm)
  lama =alam*maxnorm # Lower lambda
  lamb =blam*maxnorm # Higher lambda
  gr = (sqrt(5.0) + 1.0) / 2.0 # Golden section ratio
  toladmm = 1e-4 # Convergence tolerance for ADMM
  fc = lasso_admm(Xtrainhot, ytrain, lama, zero(Xtrainhot[1,:]),zero(Xtrainhot[1,:]),f, abscovinv,toladmm)
  lossc= 0.5*norm(Xtesthot*fc[1].-ytest)^2 # Test error for initial lower lambda
  fd = lasso_admm(Xtrainhot, ytrain, lamb, zero(Xtrainhot[1,:]),zero(Xtrainhot[1,:]),f, abscovinv,toladmm)
  lossd= 0.5*norm(Xtesthot*fd[1].-ytest)^2 # Test error for initial higher lambda
  iter = 2
  meanlam = zero(1.0:100.0)
  #meanlam[iter] = (lama+lamb)/2
  meanloss = zero(1.0:100.0)
  #meanloss[1] = max(lossc,lossd)
  #meanloss[iter] = (lossc+lossd)/2
  lamc = lamb - (lamb - lama) / gr
  lamd = lama + (lamb - lama) / gr
  println("lossc =$lossc")
  println("lossd =$lossd")
  println("lambdaa =$lama")
  println("lambdab =$lamb")
  println("lambdac =$lamc")
  println("lambdad =$lamd")
  iter = 2
  nodrun = 0
  while abs(lamc - lamd)/((lamc + lamd)/2.0) > tolgss # Run GSS until convergence
    iter = iter+1
    if iter == 3
    fc = lasso_admm(Xtrainhot, ytrain, lamc, fc[1],fc[2],f, abscovinv,toladmm)
    lossc= 0.5*norm(Xtesthot*fc[1].-ytest)^2 # Test error for initial lower lambda
    fd = lasso_admm(Xtrainhot, ytrain, lamd, fd[1],fd[2],f, abscovinv,toladmm)
    lossd= 0.5*norm(Xtesthot*fd[1].-ytest)^2 # Test error for initial higher lambda
    else
    if nodrun==1
    fc = lasso_admm(Xtrainhot, ytrain, lamc, fc[1],fc[2],f, abscovinv,toladmm)
    lossc= 0.5*norm(Xtesthot*fc[1].-ytest)^2 # Test error for initial lower lambda
    else
    fd = lasso_admm(Xtrainhot, ytrain, lamd, fd[1],fd[2],f, abscovinv,toladmm)
    lossd= 0.5*norm(Xtesthot*fd[1].-ytest)^2 # Test error for initial higher lambda
    end
    end
    meanlam[iter] = (lamc+lamd)/2.0
    meanloss[iter] = (lossc+lossd)/2.0
    # Stop GSS if test MSE is increased two consecutive iterations
    if (meanloss[iter] > meanloss[iter-1])&&(meanloss[iter-1] > meanloss[iter-2])
      break
    end
    if lossc < lossd
      lamb = lamd
      fd=fc
      lossd=lossc
      nodrun=1
    else
      lama = lamc
      fc=fd
      lossc=lossd
      nodrun=0
    end
    lamc = lamb - (lamb - lama) / gr
    lamd = lama + (lamb - lama) / gr
    #println("lossc =$lossc")
    #println("lossd =$lossd")
    println("lambdac =$lamc")
    println("lambdad =$lamd")
  end
  # Final ADMM for optimized lambda
  lamopt = meanlam[iter-2]
  fmean1 = (fc[1]+fd[1])/2.0
  fmean2 = (fc[2]+fd[2])/2.0
  fopt = lasso_admm(Xtrainhot, ytrain, lamopt, fmean1,fmean2,f, abscovinv,toladmm)
  lossopt= 0.5*norm(Xtesthot*fopt[1].-ytest)^2
  acc = cor(Xtesthot*fopt[1],ytest)

  return lamopt,fopt,lossopt,acc
end

# Read QTLMAS2010 data, and partition into train and test parts
# (given that the data file is in the working directory)
X = readdlm("QTLMAS2010ny012.csv",',')
ytot = (X[:,1].-mean(X[:,1])) # Center y to mean zero
ytrain = ytot[1:2326]
Xtest= X[2327:size(X)[1],2:size(X)[2]]
ytest = ytot[2327:size(X)[1]]
Xtrain = X[1:2326,2:size(X)[2]]

# One hot encoding training data
Xtrain0 = copy(Xtrain)
Xtrain1 = copy(Xtrain)
Xtrain2 = copy(Xtrain)
Xtrain0[Xtrain0.==1] .= 2
Xtrain0[Xtrain0.==0] .= 1
Xtrain0[Xtrain0.==2] .= 0
Xtrain1[Xtrain1.==2] .= 0
Xtrain2[Xtrain2.==1] .= 0
Xtrain2[Xtrain2.==2] .= 1
Xtrainhot = hcat(Xtrain0,Xtrain1,Xtrain2)
# Set unimportant allocations to zero
Xtrain0 = 0
Xtrain1 = 0
Xtrain2 = 0
Xtrain = 0

# One hot encoding test data
Xtest0 = copy(Xtest)
Xtest1 = copy(Xtest)
Xtest2 = copy(Xtest)
Xtest0[Xtest0.==1] .= 2
Xtest0[Xtest0.==0] .= 1
Xtest0[Xtest0.==2] .= 0
Xtest1[Xtest1.==2] .= 0
Xtest2[Xtest2.==1] .= 0
Xtest2[Xtest2.==2] .= 1
Xtesthot = hcat(Xtest0,Xtest1,Xtest2)
# Set unimportant allocations to zero
Xtest0 = 0
Xtest1 = 0
Xtest2 = 0
Xtest = 0

# Factor for initial lower lambda
alam = 0.0001
# Factor for initial upper lambda
blam = 1.0
# Convergence factor for lambda in gss_opt
tolgss = 0.01
# Find lambda where all reg coeff are zero
maxnorm = norm(Xtrainhot'*ytrain, Inf)
# The least squares loss function
f = LeastSquares(Xtrainhot, ytrain)
# Inverse covariances to be used as weights in the adaptive lasso
abscovinv = 1.0./abs.(cov(Xtrainhot,ytrain))

# Run AUTALASSO with timing
@time res = gss_opt(alam, blam, tolgss, Xtesthot, ytest,abscovinv,maxnorm)

# Save regression coefficients, lambda, MSE and ACC to text files
writedlm("outbetaQTLMAS.txt", res[2][1])
writedlm("outlambdaQTLMAS.txt", res[1])
writedlm("outMSEQTLMAS.txt", res[3]/length(ytest)*2)
writedlm("outACCQTLMAS.txt", res[4])

# Predict observations for test data
ytesthat = Xtesthot*res[2][1]
