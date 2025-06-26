	def calculate_g_cc_i(self, i):
		"""Calculates Greens function for the central with given parameters at given frequency w.

		Args:
			i: (int): frequency index

		Returns:
			g_cc (np.ndarray): Greens function for the central part
		"""

		w = self.w
		sigma = self.Sigma
		n_l = self.n_l
		n_r = self.n_r
		gamma = self.gamma
		in_plane = self.in_plane
		D = self.D
		D = copy.copy(D)

		n_atoms = int(D.shape[0] / self.dimension)

		# set up self energies
		sigma_L = np.zeros((n_atoms * self.dimension, n_atoms * self.dimension), complex)
		sigma_R = np.zeros((n_atoms * self.dimension, n_atoms * self.dimension), complex)
		if (in_plane == True):
			lower = 2
		else:
			lower = 0

		sigma_L, sigma_R = self.set_up_Sigma_matrices(i)

		#correct momentum conservation
		if(self.model == 0 or self.model == 1 or self.model == 2):
			# correct momentum conservation (for every other system)
			# convert to hartree/Bohr**2
			gamma_hb = gamma * eV2hartree / ang2bohr ** 2

			for u in range(lower, self.dimension):
				for n_l_ in n_l:
					# remove mass weighting
					K_ = D[n_l_ * self.dimension + u][n_l_ * self.dimension + u] * top.atom_weight(self.M_C)
					# correct momentum
					K_ = K_ - gamma_hb
					# add mass weighting again
					D_ = K_ / top.atom_weight(self.M_C)
					D[n_l_ * self.dimension + u][n_l_ * self.dimension + u] = D_

				for n_r_ in n_r:
					# remove mass weighting
					K_ = D[n_r_ * self.dimension + u][n_r_ * self.dimension + u] * top.atom_weight(self.M_C)
					# correct momentum
					K_ = K_ - gamma_hb
					# add mass weighting again
					D_ = K_ / top.atom_weight(self.M_C)
					D[n_r_ * self.dimension + u][n_r_ * self.dimension + u] = D_

		if(self.model == 3 or self.model == 5):
			# correct momentum conservation (This part is for 2D Ribbon)
			gamma_hb = gamma * const.eV2hartree / const.ang2bohr ** 2
			block_shape = self.scatter.N_y * 2
			for j in range(self.n_l[0], block_shape + self.n_l[0]):
				# (This can be used to couple just x-components)
				#if (j % 2 == 0):
				#	continue
				#	pass
				# remove mass weighting
				K_ = D[j, j] * top.atom_weight(self.M_C)
				# correct momentum
				K_ = K_ - gamma_hb
				# add mass weighting again
				D_ = K_ / top.atom_weight(self.M_C)
				D[j, j] = D_
			for j in range(sigma_R.shape[0] - block_shape, sigma_R.shape[0]):
				# (This can be used to couple just x-components)
				#if (j % 2 == 1):
				#	# continue
				#	pass
				# remove mass weighting
				K_ = D[j, j] * top.atom_weight(self.M_C)
				# correct momentum
				K_ = K_ - gamma_hb
				# add mass weighting again
				D_ = K_ / top.atom_weight(self.M_C)
				D[j, j] = D_
				
		# calculate greens function
		G = np.linalg.inv(self.w[i] ** 2 * np.identity(self.dimension * n_atoms) - D - sigma_L - sigma_R)
		return G

	def calculate_g_cc(self):
		"""Calculates Greens function for the central with given parameters at given frequency w.

		Args:
			i: (int): frequency index

		Returns:
			g_cc (np.ndarray): Greens function for the central part
		"""

		# calculate Greens function for central part
		g_cc_ = map(g_cc_i, self.i)
		
		for j, item in enumerate(g_cc_):
			self._G_cc[j] = item

	def calculate_T(self):
		# calculate Transmission
		T_ = map(self.calculate_T_i, self.i)
		if(self.eigenchannel == False):
			for j, item in enumerate(T_):
				self._T[j] = float(item)
		else:
			for j, (item1, item2) in enumerate(T_):
				self._T[j] = float(item1)
				self.T_channel_vals[j,:] = np.array([float(v) for v in item2])

	def calculate_T_i(self, i):
		data_path = self.data_path
		coord = self.coord
		eigenchannel = self.eigenchannel
		every_nth = self.every_nth
		channel_max = self.channel_max

		trans_prob_matrix = self.calc_trans_prob_matrix_i(i)

		if (eigenchannel == True):
			write_out = False
			energy = -1
			if (i % every_nth == 0 and every_nth != -1):
				write_out = True

			T, T_channel = self.calc_eigenchannel_i(i, write_out)
			return T, T_channel
		else:
			T = np.real(np.trace(trans_prob_matrix))
			return T

	def calc_eigenchannel(self, trans_prob_matrix, calc_path, channel_max, coord, write_out, energy):
		"""
		Calculates phonon transmission eigenchannels according to Klöckner, J. C., Cuevas, J. C., & Pauly, F. (2018). Transmission eigenchannels for coherent phonon transport. Physical Review B, 97(15), 155432 (https://doi.org/10.1103/PhysRevB.97.155432)
		Args:
			trans_prob_matrix (np.ndarray): Transmission prob matrix (eq 25 in ref)
			calc_path (String): path to calculation
			channel_max (int): number of stored eigenvaues
			coord (array): coord file loaded with top
			write_out (bool): write channel information
			energy (float): Phonon energy in meV (for filename)

		Returns: T, T_vals: Total transmission, Contribution of each channel (up to channel_max)

		"""

		eigenvalues, eigenvectors = np.linalg.eigh(trans_prob_matrix + 0*np.ones(trans_prob_matrix.shape) * (1.j * 1E-15))
		# sort eigenvalues and eigenvecors
		idx = eigenvalues.argsort()[::-1]
		eigenvalues = eigenvalues[idx]
		eigenvectors = eigenvectors[:, idx]

		def calc_displacement(z_value):
			# z_value = eigenvectors[i, j]
			z_abs = np.abs(z_value)
			if (z_abs) > 0:
				phase = np.arccos(np.real(z_value) / z_abs)
				if (np.imag(z_value) <= 0):
					phase = 2.0 * np.pi - phase
			else:
				phase = 0
			# real part
			displacement = z_abs * np.cos(phase)

			return displacement

		# calc displacement
		calc_func = np.vectorize(calc_displacement)
		displacement_matrix = calc_func(eigenvectors)
		if (write_out == True):
			if (os.path.exists(calc_path + "/eigenchannels") == False):
				os.mkdir(f"{calc_path}/eigenchannels")
			eu.write_nmd_file(f"{calc_path}/eigenchannels/eigenchannel_{energy}.nmd", coord, displacement_matrix,
							  channel_max, self.dimension)

		# calculate Transmission
		T = np.sum(eigenvalues)
		return T, eigenvalues[0:channel_max]

	def set_up_Sigma_matrices(self, i):
		"""
		Sets up self energy matrices for model calculations
		Args:
			i: index

		Returns:
			Sigma_L, Sigma_R
		"""
		
			
		

		return sigma_L, sigma_R

	def calc_trans_prob_matrix_i(self, i):
		"""
		Calculates transmission prob matrix at w[i].
		Args:
			i (int): index in frequency array

		Returns: trans_prob_matrix

		"""



		in_plane = self.in_plane
		D = self.D

		sigma_L, sigma_R = self.set_up_Sigma_matrices(i)


		Gamma_L = -2 * np.imag(sigma_L)
		Gamma_R = -2 * np.imag(sigma_R)
		trans_prob_matrix = np.dot(np.dot(Gamma_L, self.G_cc[i]), np.dot(Gamma_R, np.conj(np.transpose(self.G_cc[i]))))
		#trans_prob_matrix = np.dot(np.dot(self.G_cc[i], Gamma_L), np.dot(np.conj(np.transpose(self.G_cc[i])), Gamma_R, ))
		return trans_prob_matrix

	def calc_eigenchannel(self, E):
		"""
		Calculates Tranmssion Eigenchannel at given Energy E
		Args:
			E (float): Energy in meV

		Returns:

		"""
		#convert
		w = E / (unit2SI * h_bar / (meV2J))
		#find index
		index = np.argmin(np.abs(w-self.w))
		#prepare and calculate
		self._G_cc[index] = self.calculate_G_cc_i(index)
		self.calc_eigenchannel_i(index, write_out=True)

	def calc_eigenchannel_i(self, i, write_out):
		"""
		Calculates phonon transmission eigenchannels at w[i] according to Klöckner, J. C., Cuevas, J. C., & Pauly, F. (2018). Transmission eigenchannels for coherent phonon transport. Physical Review B, 97(15), 155432 (https://doi.org/10.1103/PhysRevB.97.155432)
		Args:
			trans_prob_matrix (np.ndarray): Transmission prob matrix (eq 25 in ref)
			calc_path (String): path to calculation
			channel_max (int): number of stored eigenvaues
			coord (array): coord file loaded with top
			write_out (bool): write channel information
			energy (float): Phonon energy in meV (for filename)

		Returns: T, T_vals: Total transmission, Contribution of each channel (up to channel_max)

		"""
		energy = np.round(self.w[i] * unit2SI * h_bar / (meV2J), 3)
		trans_prob_matrix = self.calc_trans_prob_matrix_i(i)
		#TODO: Large negative Eigenvalues
		eigenvalues, eigenvectors = np.linalg.eigh(trans_prob_matrix + 0*np.ones(trans_prob_matrix.shape) * (1.j * 1E-25))
		# sort eigenvalues and eigenvecors
		idx = eigenvalues.argsort()[::-1]
		eigenvalues = eigenvalues[idx]
		eigenvectors = eigenvectors[:, idx]

		def calc_displacement(z_value):
			# z_value = eigenvectors[i, j]
			z_abs = np.abs(z_value)
			if (z_abs) > 0:
				phase = np.arccos(np.real(z_value) / z_abs)
				if (np.imag(z_value) <= 0):
					phase = 2.0 * np.pi - phase
			else:
				phase = 0
			# real part
			displacement = z_abs * np.cos(phase)

			return displacement

		# calc displacement
		calc_func = np.vectorize(calc_displacement)
		displacement_matrix = calc_func(eigenvectors)
		if (write_out == True):
			if (os.path.exists(self.data_path + "/eigenchannels") == False):
				os.mkdir(f"{self.data_path}/eigenchannels")
			eu.write_nmd_file(f"{self.data_path}/eigenchannels/eigenchannel_{energy}.nmd", self.coord, displacement_matrix,
							  channel_max, dimensions=self.dimension)

		# calculate Transmission
		T = np.sum(eigenvalues)
		#pfusch
		eigenvalues[0] = eigenvalues[0]+eigenvalues[-1]
		return T, np.asarray(eigenvalues[0:channel_max], dtype=float)