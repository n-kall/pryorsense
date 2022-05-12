def cjs_dist(x, weights):

    # normalise weights
    weights = weights/np.sum(weights)

    # sort draws and weights
    x, w = (list(x) for x in zip(*sorted(zip(x, weights))))

    bins = x[:-1]
    binwidth = np.diff(x)

    # ecdfs
    Px = np.full(shape=len(x), fill_value=1/len(x))
    Px = np.cumsum(Px)[:-1]
    Qx = np.cumsum(w/np.sum(w))[:-1]

    # integrals of ecdfs
    Px_int = np.dot(Px, binwidth)
    Qx_int = np.dot(Qx, binwidth)

    cjs_PQ = np.nansum(binwidth * (
        Px * (np.log2(Px) -
              np.log2(0.5 * Px + 0.5 * Qx)
        ))) + 0.5 / np.log(2) * (Qx_int - Px_int)

    cjs_QP = np.nansum(
        binwidth *
        Qx * (np.log2(Qx) -
              np.log2(0.5 * Qx + 0.5 * Px)
              )) + 0.5 / np.log(2) * (Px_int - Qx_int)

    bound = Px_int + Qx_int

    return np.sqrt((cjs_PQ + cjs_QP)/bound)



def cjs_deriv(lower_diff, upper_diff, upper_alpha):

    ## second-order centered difference approximation
    ## f''(x) = (f(x + dx) - 2f(x) + f(x - dx)) / dx^2
    logdiffsquare = 2 * np.log2(upper_alpha)
    grad = (upper_diff + lower_diff) / logdiffsquare

    return grad


def powerscale_sensitivity(data, variable, component, delta=0.01):

    draws = data.posterior.stack(draws=("chain", "draw"))[variable].values

    lower_w = np.exp(powerscale_lw(data=data, alpha=1/(1+delta), component=component))
    upper_w = np.exp(powerscale_lw(data=data, alpha=1+delta, component=component))

    lower_w = lower_w/np.sum(lower_w)
    upper_w = upper_w/np.sum(upper_w)


    lower_cjs = cjs_dist(x=draws, weights=lower_w)
    upper_cjs = cjs_dist(x=draws, weights=upper_w)

    logdiffsquare = 2 * np.log2(1 + delta)
    grad = (lower_cjs + upper_cjs) / logdiffsquare

    return lower_cjs, upper_cjs, grad
