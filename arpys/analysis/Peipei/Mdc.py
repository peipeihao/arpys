import xarray as xr
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import latex

@xr.register_dataarray_accessor("mdc")
class Mdc:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self.init_params = None  #Parameter for the first slice of fit
        self.init_mdc = None     #The first slice of MDC
        self.energies = []  #Stores the binding energy values of the group fit
        self.results = []   #Stores the group fitting results
        self.model=None     #Records the model used for the group fitting

    def nearest_index(self, data, value):
        abs_diff = np.abs(data - value)
        index = np.argmin(abs_diff)
        return index

    def get_mdc(self, E_val):
        '''
        returns a single slice of MDC at the binding energy nearest to 'E_val'
        '''
        return self._obj.sel({'binding':E_val},method='nearest')

    def get_result(self, E_val):
        '''
        returns the fitting result of the MDC at the closest binding energy to 'E_val'
        '''
        if not self.results:
            warnings.warn("Warning: Your 'results' list is empty, please try some fittings first.")
        energies = self.energies
        results = self.results
        E_index = self.nearest_index(energies, E_val)
        return results[E_index]

    def prep_energies_forward(self, low_E, high_E):
        data_tofit = self._obj.sel(binding=slice(low_E, high_E))
        energies = data_tofit.binding.values
        return energies
        
    def prep_energies_backward(self, low_E, high_E):
        data_tofit = self._obj.sel(binding=slice(high_E, low_E, -1))
        energies = data_tofit.binding.values
        return energies

    def single_fit(self, model, params, E_val, method='least_squares'):
        mdc = self.get_mdc(E_val)
        ydata = mdc.values
        xdata = mdc.kx.values
        result = model.fit(ydata, params, x=xdata, method=method)
        return result

    def forward_group_fit(self, model, params, low_E, high_E, method='least_squares'):
        energies = self.prep_energies_forward(low_E, high_E)
        results = []
        for E_value in energies:
            print(E_value,end='\r')
            result = self.single_fit(model, params, E_value, method=method)
            params = result.params
            results.append(result)
        return results

    def backward_group_fit(self, model, params, low_E, high_E, method='least_squares'):
        energies = self.prep_energies_backward(low_E, high_E)
        results = []
        for E_value in energies:
            print(E_value,end='\r')
            result = self.single_fit(model, params, E_value, method=method)
            params = result.params
            results.append(result)
        return results
    
    def group_fit(self, model, params, low_E, high_E, init_E, method='least_squares'):
        fw_results = self.forward_group_fit(model=model, params=params, low_E=init_E, high_E=high_E, method=method)
        bw_results = self.backward_group_fit(model=model, params=params, low_E=low_E, high_E=init_E, method=method)
        reversed_bw_results = bw_results[::-1]
        self.energies = self.prep_energies_forward(low_E, high_E)
        self.results = reversed_bw_results + fw_results
        self.model = model

    def df_fitreport(self, attr_list=['success', 'lmdif_message', 'chisqr', 'redchi']):
        '''
        This function generates a DataFrame report of the fitting attributes listed in 'attr_list'
        and indexes each row with the corresponding binding energy of the MDC slice
        :param 'attr_list' contains any wanted attributes of the ModelResult object
        '''
        results = self.results
        energies = self.energies
        dict = {}
        for attr_name in attr_list:
            attr_E = []  # the list to hold attr_vs_energy values
            for i in range(len(energies)):
                attr = getattr(results[i], attr_name)
                attr_E.append(attr)
            dict[attr_name] = attr_E
        df = pd.DataFrame(dict, index=energies)
        return df

    def df_params(self):
        '''
        This function generates a DataFrame collection of the best_values for each fitted parameter
        '''
        model = self.model
        energies = self.energies
        results = self.results
        dict = {}
        param_names = model.param_names
        for param in param_names:
            param_T = []  # the list to hold attr_vs_temperature values
            for i in range(len(energies)):
                param_instance = results[i].params.get(param)
                best_value = param_instance.value
                param_T.append(best_value)
            dict[param] = param_T
        df = pd.DataFrame(dict, index=energies)
        return df

    def plot_dispersion(self, names):
        '''
        plots the spectrum overlaid with the extracted MDC-peaks
        :param 'names' contains the names of the peak centers to plot
        '''
        data = self._obj
        fig, ax = plt.subplots()
        data.plot(x='kx', y='binding', ax=ax, cmap='twilight')
        df = self.df_params()
        for name in names:
            ax.scatter(df[name], df.index, marker='+', color='orangered')
        ax.set_xlabel('$\mathbf{k}$ ($\mathbf{\AA^{-1}}$)', fontsize=12)
        ax.set_ylabel('E-E$\mathbf{_F}$(eV)', fontsize=12, fontweight='bold')
        
    def plot_fit(self, E_val):
        '''
        plots the fit result of the mdc at E = Eval
        '''
        result = self.get_result(E_val)
        result.plot()

    def plot_components(self, E_val):
        '''
        plots each components of a fit(if the fitting model is a CompositeModel) from a group of the fits
        '''
        result = self.get_result(E_val)
        components = result.eval_components()
        names = [] # name of each component function
        fit_arrs = []
        for (key,arr) in components.items():
            names.append(key)
            fit_arrs.append(arr)
        mdc = self.get_mdc(E_val)
        xdata = mdc.kx.values
        ydata = mdc.values
        fig, ax = plt.subplots()
        ax.scatter(xdata, ydata, label='data')  # The MDC data
        ax.plot(xdata, result.best_fit, label='best_fit')  # The fit
        for i in range(len(components)):
            comp = fit_arrs[i]
            ax.plot(xdata, comp, linestyle='--',  label=f'fit_{names[i]}')
        ax.legend()

    def plot_waterfall(self, N=10, init=0, text=True):
        '''
        plots the MDC stacks overlaid with fits
        :param 'N' is the number of MDC slices to include
        :param 'init' is the first slice to plot, be default is the MDC at the lowest energy of the group fit
        :param 'text' can be used to switch off the printed texts, in case they cause s mess in the plot
        '''
        data = self._obj
        fig, ax = plt.subplots(figsize=(7, 6))
        results = self.results
        energies = self.energies
        num_of_plots = N
        tot_slices = len(results)
        distance = tot_slices // num_of_plots
        offset = 0  # offset of the first MDC slice
        for i in range(num_of_plots):
            init_index = init  # Index of the first MDC slice to plot
            index = init_index + i * distance
            xdata = data.kx.values
            ydata = results[index].data
            best_fit = results[index].best_fit
            ax.scatter(xdata, ydata + offset, color='royalblue')
            ax.plot(xdata, best_fit + offset, linestyle='-', color='darkorange')
            middle_index = len(xdata) // 2 # Just a trick to adjust the location of the text
            if text is True:
                ax.text(x=-0.1+xdata[middle_index], y=ydata[middle_index] + offset + 0.15, s=f'E={round(energies[index], 3)}eV')
            offset += ydata.max()
        ax.set_xlabel('$\mathbf{k}$ ($\mathbf{\AA^{-1}}$)', fontsize=13)



