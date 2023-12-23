from typing import Tuple
import torch

one = torch.ones((1, 1024)).cuda()

class Descriptive_MF(torch.nn.Module):
    def __init__(self, n_users: int, m_items: int, latent_dim: int):
        super(Descriptive_MF, self).__init__()
        self.latent_dim = latent_dim
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=m_items, embedding_dim=self.latent_dim
        )
        self.descriptive = torch.nn.Linear(1024, 1)
        torch.nn.init.normal_(self.user_embedding.weight, std=0.01)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.01)
        torch.nn.init.normal_(self.descriptive.weight, std=0.01)

    def get_user_regularization(self, user_id):
        return torch.sum(self.user_embedding(user_id) ** 2, dim=1)

    def get_item_regularization(self, item_id):
        return torch.sum(self.item_embedding(item_id) ** 2, dim=1) + torch.sum(self.descriptive(one) ** 2, dim=1)

    def forward(self, user_id, item_id, item_descriptive):
        users_emb = self.user_embedding(user_id)
        items_emb = self.item_embedding(item_id)
        scores = torch.sum(users_emb * items_emb, dim=1) + self.descriptive(item_descriptive).reshape(-1)
        return scores

class EmbeddingBackend(torch.nn.Module):
    '''
    Embedding Backend for collaborative filtering
    '''
    def __init__(self, n_users: int, m_items: int, latent_dim: int):
        super(EmbeddingBackend, self).__init__()
        self.latent_dim = latent_dim
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=m_items, embedding_dim=self.latent_dim
        )
        torch.nn.init.normal_(self.user_embedding.weight, std=0.01)
        torch.nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_id, item_id):
        return self.user_embedding(user_id), self.item_embedding(item_id)


class MLPBase(torch.nn.Module):
    '''
    Implementation of MLP Matrix Factorization
    https://arxiv.org/abs/1708.05031
    '''
    def __init__(self, latent_dim: int, hidden_dims: Tuple[int, ...] = (64, 32)):
        super(MLPBase, self).__init__()
        hidden_layers = []
        dims = [*hidden_dims, latent_dim]
        for i in range(len(dims) - 1):
            hidden_layers.append(torch.nn.Dropout(0.1))
            hidden_layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            hidden_layers.append(torch.nn.ReLU())
        self.mlp_dense = torch.nn.Sequential(*hidden_layers)

    def forward(self, user_embed, item_embed):
        vector = self.mlp_dense(user_embed * item_embed)
        return vector

class MF(torch.nn.Module):
    '''
    Conventional Matrix Factorization
    '''
    def __init__(self, n_users: int, m_items: int, latent_dim: int):
        super(MF, self).__init__()
        self.embed = EmbeddingBackend(n_users, m_items, latent_dim)

    def forward(self, user_id, item_id):
        user_embed, item_embed = self.embed(user_id, item_id)
        scores = torch.sum(user_embed * item_embed, dim=1)
        return scores

class GMF(torch.nn.Module):
    '''
    Implementation of Generalized Matrix Factorization
    https://arxiv.org/abs/1708.05031
    '''
    def __init__(self, n_users: int, m_items: int, latent_dim: int):
        super(GMF, self).__init__()
        self.embed = EmbeddingBackend(n_users, m_items, latent_dim)
        self.gmf_dense = torch.nn.Linear(latent_dim, 1)

    def forward(self, user_id, item_id):
        user_embed, item_embed = self.embed(user_id, item_id)
        return self.gmf_dense(user_embed * item_embed).reshape(-1)


class MLP_MF(torch.nn.Module):
    '''
    Implementation of MLP Matrix Factorization
    https://arxiv.org/abs/1708.05031
    '''
    def __init__(self, n_users: int, m_items: int, latent_dim: int, hidden_dims: Tuple[int, ...] = (64, 32)):
        super(MLP_MF, self).__init__()
        self.embed = EmbeddingBackend(n_users, m_items, latent_dim)
        self.mlp = MLPBase(latent_dim, hidden_dims)
        self.dense = torch.nn.Linear(latent_dim, 1)

    def forward(self, user_id, item_id):
        return self.dense(self.mlp(*self.embed(user_id, item_id))).reshape(-1)

class NeuMF(torch.nn.Module):
    '''
    Implementation of Neural Matrix Factorization
    https://arxiv.org/abs/1708.05031
    '''
    def __init__(self, n_users: int, m_items: int, latent_dim: int, hidden_dims: Tuple[int, ...] = (20, 10)):
        super(NeuMF, self).__init__()
        self.gmf_embed = EmbeddingBackend(n_users, m_items, latent_dim)
        self.mlp_embed = EmbeddingBackend(n_users, m_items, hidden_dims[0])
        self.mlp = MLPBase(latent_dim, hidden_dims)
        self.dense = torch.nn.Linear(2 * latent_dim, 1)

    def forward(self, user_id, item_id):
        gmf_vec = torch.mul(*self.gmf_embed(user_id, item_id))
        mlp_vec = self.mlp(*self.mlp_embed(user_id, item_id))
        return self.dense(torch.concat([gmf_vec, mlp_vec], dim=-1)).reshape(-1)


class LightGCN(torch.nn.Module):
    def __init__(self, n_users: int, m_items: int, latent_dim: int, layers: int):
        super(LightGCN, self).__init__()
        self.latent_dim = latent_dim
        self.layers = layers
        
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=m_items, embedding_dim=self.latent_dim
        )

class NCF_GitHub(torch.nn.Module):
	def __init__(self, user_num, item_num, factor_num, num_layers,
					dropout=0.1, model='NeuMF-end', GMF_model=None, MLP_model=None):
		super(NCF_GitHub, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors;
		num_layers: the number of layers in MLP model;
		dropout: dropout rate between fully connected layers;
		model: 'MLP', 'GMF', 'NeuMF-end', and 'NeuMF-pre';
		GMF_model: pre-trained GMF weights;
		MLP_model: pre-trained MLP weights.
		"""		
		self.dropout = dropout
		self.model = model
		self.GMF_model = GMF_model
		self.MLP_model = MLP_model

		self.embed_user_GMF = torch.nn.Embedding(user_num, factor_num)
		self.embed_item_GMF = torch.nn.Embedding(item_num, factor_num)
		self.embed_user_MLP = torch.nn.Embedding(
				user_num, factor_num * (2 ** (num_layers - 1)))
		self.embed_item_MLP = torch.nn.Embedding(
				item_num, factor_num * (2 ** (num_layers - 1)))

		MLP_modules = []
		for i in range(num_layers):
			input_size = factor_num * (2 ** (num_layers - i))
			MLP_modules.append(torch.nn.Dropout(p=self.dropout))
			MLP_modules.append(torch.nn.Linear(input_size, input_size//2))
			MLP_modules.append(torch.nn.ReLU())
		self.MLP_layers = torch.nn.Sequential(*MLP_modules)

		if self.model in ['MLP', 'GMF']:
			predict_size = factor_num 
		else:
			predict_size = factor_num * 2
		self.predict_layer = torch.nn.Linear(predict_size, 1)

		self._init_weight_()

	def _init_weight_(self):
		""" We leave the weights initialization here. """
		if not self.model == 'NeuMF-pre':
			torch.nn.init.normal_(self.embed_user_GMF.weight, std=0.01)
			torch.nn.init.normal_(self.embed_user_MLP.weight, std=0.01)
			torch.nn.init.normal_(self.embed_item_GMF.weight, std=0.01)
			torch.nn.init.normal_(self.embed_item_MLP.weight, std=0.01)

			for m in self.MLP_layers:
				if isinstance(m, torch.nn.Linear):
					torch.nn.init.xavier_uniform_(m.weight)
			torch.nn.init.kaiming_uniform_(self.predict_layer.weight, 
									a=1, nonlinearity='sigmoid')

			for m in self.modules():
				if isinstance(m, torch.nn.Linear) and m.bias is not None:
					m.bias.data.zero_()
		else:
			# embedding layers
			self.embed_user_GMF.weight.data.copy_(
							self.GMF_model.embed_user_GMF.weight)
			self.embed_item_GMF.weight.data.copy_(
							self.GMF_model.embed_item_GMF.weight)
			self.embed_user_MLP.weight.data.copy_(
							self.MLP_model.embed_user_MLP.weight)
			self.embed_item_MLP.weight.data.copy_(
							self.MLP_model.embed_item_MLP.weight)

			# mlp layers
			for (m1, m2) in zip(
				self.MLP_layers, self.MLP_model.MLP_layers):
				if isinstance(m1, nn.Linear) and isinstance(m2, nn.Linear):
					m1.weight.data.copy_(m2.weight)
					m1.bias.data.copy_(m2.bias)

			# predict layers
			predict_weight = torch.cat([
				self.GMF_model.predict_layer.weight, 
				self.MLP_model.predict_layer.weight], dim=1)
			precit_bias = self.GMF_model.predict_layer.bias + \
						self.MLP_model.predict_layer.bias

			self.predict_layer.weight.data.copy_(0.5 * predict_weight)
			self.predict_layer.bias.data.copy_(0.5 * precit_bias)

	def forward(self, user, item):
		if not self.model == 'MLP':
			embed_user_GMF = self.embed_user_GMF(user)
			embed_item_GMF = self.embed_item_GMF(item)
			output_GMF = embed_user_GMF * embed_item_GMF
		if not self.model == 'GMF':
			embed_user_MLP = self.embed_user_MLP(user)
			embed_item_MLP = self.embed_item_MLP(item)
			interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
			output_MLP = self.MLP_layers(interaction)

		if self.model == 'GMF':
			concat = output_GMF
		elif self.model == 'MLP':
			concat = output_MLP
		else:
			concat = torch.cat((output_GMF, output_MLP), -1)

		prediction = self.predict_layer(concat)
		return prediction.view(-1)