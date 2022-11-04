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

#A helper function for extracting model parameters
function extract_params(model)
    params = Flux.params(model) #Extract the parameters in Flux form
    final_params = vec(params[1]) #Extract the weights from the first layer and flatten their matrix
    for i in 2:length(params) #Iteratively add the flattened parameters to the previously defined vector
        final_params = vcat(final_params, vec(params[i]))
    end
    return final_params #Return the final vector of parameters
end

println("[> Loaded $(@__FILE__)")