""" Create HAR regressors """
function HAR_regressors(data0)

    RVd = data0[22:end-1] # days
    RVw = zeros(length(RVd)) # weeks
    for i in 22:(length(data0)-1)
       temp = 0
       for h in 0:4
           temp = temp + data0[i-h]
       end
       RVw[i-21]= temp / 5
    end
    
    RVm = zeros(length(RVd)) # months
    for i in 22:(length(data0)-1) 
       temp = 0
       for h in 0:21
           temp = temp + data0[i-h]
       end
       RVm[i-21] = temp / 22;
    end

    return [RVd RVw RVm]
end

"""Train Test split"""
#very basic implemenattion of train test splitting
function train_test_split(predictors, target, test_size)
    break_point = floor(Int, size(predictors)[1]*(1-test_size))
    X_train = predictors[1:break_point,:] |> permutedims .|> Float32
    X_test =  predictors[break_point+1:end,:] |> permutedims .|> Float32

    y_train = target[1:break_point,:] |> permutedims .|> Float32
    y_test = target[break_point+1:end,:] |> permutedims .|> Float32
    
    return X_train, X_test, y_train, y_test
    
end

#Function calculating and plotting the AutoCorrelation Function (ACF)
function plot_ACF(time_series, title)
    Plots.plot(StatsBase.autocor(time_series), title = title, label = "ACF")
    Plots.hline!([1.96 / sqrt(length(time_series))], c = :red, linestyle = :dash, label = "95% confidence level")
    Plots.hline!([-1.96 / sqrt(length(time_series))], c = :red, linestyle = :dash, label = false)
end

#Function standardizing a given data set based on mean and standard deviation of the training set
function standardize(data) #Define a function normalizing data based on mean and std of the training set
    data1 = copy(data) #Make a copy of the argument to prevent the function from changing it in place
    for i in 1:length(mean_X_train) #Loop through the rows and normalize them
        data1[i,:] = (data1[i,:] .- mean_X_train[i]) / std_X_train[i]
    end
    return data1
end

#Function calculating the loss of a HAR model
function loss_HAR((beta_0, beta_1, beta_2, beta_3)) #Define the loss function for HAR
    return StatsBase.mean((beta_0 .+ beta_1 * X_train[1, :] + beta_2 * X_train[2, :] + beta_3 * X_train[3, :] - transpose(y_train)) .^2)
end

#Function calculating the fitted values of the HAR model
function calc_fitted(coefs, data)
    return coefs[1] .+ coefs[2] * data[1, :] + coefs[3] * data[2, :] + coefs[4] * data[3, :]
end

#Function calculating the Mean Squared Error (MSE)
function MSE(fitted, actual)
    return StatsBase.mean((fitted - actual) .^ 2)
end

#A helper function for extracting model parameters from a neural network
function extract_params(model)
    params = Flux.params(model) #Extract the parameters in Flux form
    final_params = vec(params[1]) #Extract the weights from the first layer and flatten their matrix
    for i in 2:length(params) #Iteratively add the flattened parameters to the previously defined vector
        final_params = vcat(final_params, vec(params[i]))
    end
    return final_params #Return the final vector of parameters
end

#A function for Neural Network estimation. We assume only a single hidden layer since it should be sufficient for approximation of any function
function train_NN(X_train, y_train, X_test, y_test; dropout = false, nodes = 5, activ_func = Flux.relu, output_func = Flux.identity, loss_func = Flux.Losses.mse, α = 0, return_inits = true, learn_rate = 0.001, opt = Flux.Descent, batch_size = false, n_epochs = 1000, seed = 420, verbose = true)
    Random.seed!(seed) #Set the seed for reproducibility
    if dropout == false
        model = Flux.Chain(Flux.Dense(size(X_train, 1), nodes, activ_func), Flux.Dense(nodes, size(y_train, 1), output_func)) #Specify the model
    else
        model = Flux.Chain(Flux.Dense(size(X_train, 1), nodes, activ_func), Flux.Dropout(dropout), Flux.Dense(nodes, size(y_train, 1), output_func)) #Specify the model
    end
    l2_norm(x) = sum(abs2, x) #Define a function to calculate the L2 norm
    loss(x, y) = loss_func(model(x), y) + α*sum(l2_norm, Flux.params(model)) #Specify the loss function with an L2 regularization term (defaultly, regularization is disabled by setting α = 0)
    parameters = Flux.params(model) #Specify the parameters to be estimated
    if return_inits #If required, save the initial parameters
        initial_parameters = extract_params(model)
    end
    if batch_size == false #In case all observations are to be used for each epoch
        data_train = [(X_train, y_train)]
    else
        data_train = Flux.DataLoader((X_train, y_train), batchsize = batch_size) #Dissect the training data into mini-batches
    end
    previous_loss = loss_func(model(X_test), y_test) #Initialize the loss for comparison in the early stopping condition
    for epoch in 1:n_epochs #Train the model iteratively
        if epoch % 10 == 0 #Each 10 epochs check the early stopping condition
            current_loss = loss_func(model(X_test), y_test) #Store the current test loss
            if current_loss >= previous_loss #If the test error increased or stayed the same, stop optimalization
                if verbose
                    println("Early stopping at epoch $epoch \t MSE (train): ", loss(X_train, y_train), " \t MSE (test): ", current_loss)
                end
                break #Stop optimalization
            end
            previous_loss = copy(current_loss) #If the condition was not met, store the current loss for future comparison
        end
        Flux.train!(loss, parameters, data_train, opt(learn_rate)) #Train the model for the current epoch
        if verbose
            epoch % (n_epochs / 10) == 0 ? println("Epoch $epoch \t MSE (train): ", loss(X_train, y_train), " \t MSE (test): ", loss_func(model(X_test), y_test)) : nothing #Report the losses for each tenth of the number of epochs
        end
    end
    if return_inits #If required, return the initial parameters
        return initial_parameters, model
    else #Otherwise return only the trained model
        return model
    end
end;

#A function performing a grid search over a specified parameter grid
function gridSearch(param_grid, X_train, y_train, X_test, y_test; verbose = true, return_best = false, func = train_NN)
    param_grid_keys = collect(keys(param_grid)) #Extract the names of the specified parameters
    results = DataFrames.DataFrame(Iterators.product(values(param_grid)...)) #Construct a data frame from all combinations of the parameter grid
    DataFrames.rename!(results, param_grid_keys)
    results.MSE = zeros(size(results)[1]) #Specify an enmpty column for the mSE of each model
    if return_best #In case we want the best model returned
        models = Vector{Any}(undef, size(results)[1]) #Initialize an empty vector to store the models
    end
    for row in 1:size(results)[1] #Iterate over all possible combinations
        model = func(X_train, y_train, X_test, y_test; Dict(param_grid_keys[i] => results[row, i] for i in 1:length(param_grid_keys))..., return_inits = false, verbose = false) #Train a model with the specified parameters
        if return_best
            models[row] = model
        end
        results[row, :MSE] = Flux.Losses.mse(model(X_test), y_test)
        if verbose
            println("Iteration $row out of ", size(results)[1], " complete")
        end
    end
    if return_best
        return results, models[argmin(results.MSE)] #Return the final table along with the best model
    else
        return results #Otherwise return only the table of results
    end
end;

#A function for Recurrent Neural Network estimation. We assume only a single hidden layer since it should be sufficient for approximation of any function
function train_RNN(X_train, y_train, X_test, y_test; dropout = false, nodes = 5, activ_func = Flux.relu, output_func = Flux.identity, loss_func = Flux.Losses.mse, α = 0, return_inits = true, learn_rate = 0.001, opt = Flux.Descent, n_epochs = 100, seed = 420, verbose = true, max_patience = 10)
    Random.seed!(seed) #Set the seed for reproducibility
    if dropout == false
        model = Flux.Chain(Flux.RNN(size(X_train, 1), nodes, activ_func), Flux.Dense(nodes, size(y_train, 1), output_func)) #Specify the model
    else
        model = Flux.Chain(Flux.RNN(size(X_train, 1), nodes, activ_func), Flux.Dropout(dropout), Flux.Dense(nodes, size(y_train, 1), output_func)) #Specify the model
    end
    l2_norm(x) = sum(abs2, x) #Define a function to calculate the L2 norm
    loss(x, y) = loss_func(model(x), y) + α*sum(l2_norm, Flux.params(model)) #Specify the loss function with an L2 regularization term (defaultly, regularization is disabled by setting α = 0)
    parameters = Flux.params(model) #Specify the parameters to be estimated
    if return_inits #If required, save the initial parameters
        initial_parameters = extract_params(model)
    end
    data_train = [(X_train, y_train)] #Specify the training data    
    best_loss = Inf #Initialize the best loss
    best_model = deepcopy(model) #Initialize the best model
    patience = 0 #Initialize patience count
    for epoch in 1:n_epochs #Train the model iteratively
        Flux.reset!(model) #Reset hidden state every epoch before training since we begin at t=1
        Flux.train!(loss, parameters, data_train, opt(learn_rate)) #Train model for the current epoch
        current_test_loss = loss_func(model(X_test), y_test) #Calculate the current test loss without reseting the model
        Flux.reset!(model) #Reset the model to run it through the train data
        current_train_loss = loss(X_train, y_train) #Store current training loss
        if best_loss > current_test_loss #Check if the current model is better than the best one so far
            best_model = deepcopy(model) #If so, store it
            best_loss = copy(current_test_loss) #And its loss as well
        else #Otherwise indicate that the model is worse
            patience += 1
        end
        if verbose
            epoch % (n_epochs / 10) == 0 ? println("Epoch $epoch \t MSE (train): ", current_train_loss, " \t MSE (test): ", current_test_loss) : nothing #Report the losses for each tenth of the number of epochs
        end
        if patience >= max_patience #Check if the patience run out
            if verbose
                println("Early stopping at epoch $epoch. \t Final MSE (test): ", best_loss)
            end
            break #Stop optimalization
        end
    end
    if return_inits #If required, return the initial parameters
        return initial_parameters, best_model
    else #Otherwise return only the trained model
        return best_model
    end
end;

#Define function performing ensemble averaging for a set of NN models
function get_ensemble(X_train, y_train, X_test, y_test, models; weights = false, loss = Flux.Losses.mse, print_models_score = true)
    #parameters: X_train, y_train, X_test, y_test: data used for training the NNs
    #            models: list containing trained NNs (product of train_NN() function)
    #optional:   weights: list of integers or floats of same length as models
    #            loss: loss function to be used for evaluating result
    fitted_ensemble = zeros(Float64, 1, length(y_train)) #initialize empty matrices for ensembled values
    preds_ensemble = zeros(Float64, 1, length(y_test))

    if print_models_score
        println("Score of each model from the ensemble:")
        for NN in models
            println("MSE (train): ", loss(NN(X_train), y_train), " \t MSE (test): ", loss(NN(X_test), y_test))
        end
    end
 
    if weights == false #get weights of each model to be averaged
       weights = ones(length(models)) ./length(models) #if no weights given, produce simple average (same weights for all)
    else  
       weights = weights ./sum(weights) #(transform vector of weights such that it sums to 1)
    end
 
    i = 1
    for NN in models #for each model, get fitted/predicted values, multiply by weight and add them to the ensemble
       fitted_ensemble = fitted_ensemble .+ (NN(X_train)) .*weights[i]
       preds_ensemble = preds_ensemble .+ (NN(X_test)) .*weights[i]
       i +=1
    end      
 
    println("Score of the ensemble:")
    println("MSE (train): ", loss(fitted_ensemble, y_train), " \t MSE (test): ", loss(preds_ensemble, y_test))
    return [fitted_ensemble, preds_ensemble]
 end;

#Ensemble averaging for RNNs
function get_ensemble_RNN(X_train, y_train, X_test, y_test, models; weights = false, loss = Flux.Losses.mse, print_models_score = true)
    #parameters: X_train, y_train, X_test, y_test: data used for training the NNs
    #            models: list containing trained NNs (product of train_NN() function)
    #optional:   weights: list of integers or floats of same length as models
    #            loss: loss function to be used for evaluating result
    fitted_ensemble = zeros(Float64, 1, length(y_train)) |> permutedims #initialize empty matrices for ensembled values
    preds_ensemble = zeros(Float64, 1, length(y_test)) |> permutedims

    if print_models_score
        println("Score of each model from the ensemble:")
        for NN in models
            println("MSE (train): ", loss(NN(X_train)[1,1,:], y_train[1,1,:]), " \t MSE (test): ", loss(NN(X_test)[1,1,:], y_test[1,1,:]))
        end
    end
 
    if weights == false #get weights of each model to be averaged
       weights = ones(length(models)) ./length(models) #if no weights given, produce simple average (same weights for all)
    else  
       weights = weights ./sum(weights) #(transform vector of weights such that it sums to 1)
    end
 
    i = 1
    for NN in models #for each model, get fitted/predicted values, multiply by weight and add them to the ensemble
       fitted_ensemble = fitted_ensemble .+ (NN(X_train)[1,1,:]) .*weights[i]
       preds_ensemble = preds_ensemble .+ (NN(X_test)[1,1,:]) .*weights[i]
       i +=1
    end      
 
    println("Score of the ensemble:")
    println("MSE (train): ", loss(fitted_ensemble, y_train[1,1,:]), " \t MSE (test): ", loss(preds_ensemble, y_test[1,1,:]))
    return [fitted_ensemble, preds_ensemble]
 end; 

##helper function to get sample size and return indicies ranges to be used further in CV
function get_timesplits(sample_size; splits=4, test_size = 0.8) #parameters: Int, Int, float
    foldsize = floor(Int, (sample_size/splits)) #compute size of one fold
    fold_test_size = floor(Int, (foldsize*test_size)) #compute size of testing set in each fold
    ranges = [(foldsize*i-foldsize+1):(foldsize*i) for i in 1:splits] #get ranges of folds (e.g. 1:100)
    timesplits = [[range[1]:range[1]+fold_test_size, range[1]+fold_test_size+1:range[end]] for range in ranges] #get ranges of train/test splits (e.g. [1:81, 82:100])
    return timesplits
end;

##Function performing the Cross-Validation of specified NN
function cross_validate(X_train, y_train; folds = 4, test_size = 0.8, loss = Flux.Losses.mse, get_models = false,
                        dropout = false, nodes = 5, activ_func = Flux.relu, output_func = Flux.identity, 
                        loss_func = Flux.Losses.mse, α = 0, return_inits = false, learn_rate = 0.001, 
                        opt = Flux.Descent, batch_size = false, n_epochs = 200, seed = 420, verbose = true)
    #X and y need to be specified (assumed X_train, y_train) from before; 
    #other params: n. of folds, test size to be used in CV,loss function to be used, and all params of train_NN()
    timesplits = get_timesplits(size(X_train, 2), splits = folds, test_size = test_size)

    mses_ls = []
    models = []
    for k in 1:folds
        #split the data on training and testing , target on training and testing
        train_range = timesplits[k][1] #get the range of X for current fold
        test_range = timesplits[k][2] #get the range of y for current fold

        X_fold_train = X_train[:,train_range] #get current fold training X and y training
        y_fold_train = y_train[:, train_range]

        X_fold_test = X_train[:,test_range] #get current fold training X and y training
        y_fold_test = y_train[:, test_range]        

        #train the network
        NN_fold = train_NN(X_fold_train, y_fold_train, X_fold_test, y_fold_test; 
                                            dropout = dropout, nodes = nodes, n_epochs = n_epochs, 
                                            learn_rate = learn_rate, activ_func = activ_func, 
                                            return_inits = return_inits, output_func = output_func, 
                                            loss_func = loss_func, α = α, opt = opt, batch_size = batch_size,
                                            seed = seed, verbose = verbose);
        append!(models, NN_fold)
        
        #get the mse
        mse = loss(NN_fold(X_fold_test), y_fold_test)
        append!( mses_ls, mse ) #append MSE
        println("Fold $k/$folds, Out-of-sample MSE: ", mse)
    end
    #avergae the mses
    println("Cross-validation done. Average MSE: ", StatsBase.mean(mses_ls))

    if get_models
        return mses_ls, models
    else
        return mses_ls
    end
end;

#Function computing Realized Variance from kibot intraday data
function get_RV_from_kibot(data_OIH, returns_log = false)
    data_close = data_OIH[:,[1,6]] #saving only dates and close adjusted price
    dates = unique(data_OIH[:,1]) #saving vector of dates
    RV_OIH = []
    for date in dates
        day_prices = data_close[:,2][data_close[:,1] .== date]

        if returns_log  #geting returns or logreturns
            returns = [log(fract) for fract in day_prices[2:end] ./ day_prices[1:end-1]] #returns: (r_i - r_i-1) / r_i-1
        else 
            returns = (day_prices[2:end] .- day_prices[1:end-1])./day_prices[1:end-1] #logreturns: log(r_i / r_i-1)
        end

        RV = sum(returns .^2) #RV as sum of squared intraday rets
        append!( RV_OIH, RV )
    end

    return RV_OIH
end

println("[> Loaded $(@__FILE__)")


