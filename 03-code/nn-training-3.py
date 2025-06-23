import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import seaborn as sns

class NeuralNetworkVisualizer:
    def __init__(self):
        # Architettura della rete: 2 input -> 4 hidden -> 2 output
        self.input_size = 2
        self.hidden_size = 4
        self.output_size = 2
        
        # Inizializzazione pesi casuali
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.5
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.5
        self.b2 = np.zeros((1, self.output_size))
        
        # Parametri di apprendimento
        self.learning_rate = 0.1
        
        # Generazione dataset
        self.generate_dataset()
        
        # Setup figura
        self.setup_figure()
        
        # Traccia dell'addestramento
        self.training_history = []
        self.current_sample = 0
        
    def generate_dataset(self, n_samples=100):
        """Genera un dataset di classificazione binaria"""
        np.random.seed(42)
        
        # Genera due cluster
        cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n_samples//2)
        cluster2 = np.random.multivariate_normal([-1, -1], [[0.5, 0], [0, 0.5]], n_samples//2)
        
        self.X = np.vstack([cluster1, cluster2])
        self.y = np.hstack([np.ones(n_samples//2), np.zeros(n_samples//2)])
        
        # Normalizzazione
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        
        # Converti labels in one-hot encoding
        self.y_onehot = np.zeros((len(self.y), 2))
        self.y_onehot[np.arange(len(self.y)), self.y.astype(int)] = 1
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward_pass(self, X):
        """Forward pass della rete neurale"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward_pass(self, X, y_true, y_pred):
        """Backward pass e aggiornamento pesi"""
        m = X.shape[0]
        
        # Gradiente output layer
        dz2 = y_pred - y_true
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Gradiente hidden layer
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.a1 * (1 - self.a1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Aggiornamento pesi
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def setup_figure(self):
        """Setup della figura per l'animazione"""
        self.fig = plt.figure(figsize=(16, 10))
        
        # Layout con 3 subplot
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
        
        # Subplot per la rete neurale
        self.ax_network = self.fig.add_subplot(gs[:, 0])
        self.ax_network.set_xlim(-0.5, 3.5)
        self.ax_network.set_ylim(-0.5, 4.5)
        self.ax_network.set_aspect('equal')
        self.ax_network.set_title('Rete Neurale\n(2→4→2)', fontsize=14, fontweight='bold')
        self.ax_network.axis('off')
        
        # Subplot per i dati
        self.ax_data = self.fig.add_subplot(gs[0, 1])
        self.ax_data.set_title('Dataset e Classificazione', fontsize=12, fontweight='bold')
        
        # Subplot per la loss
        self.ax_loss = self.fig.add_subplot(gs[1, 1])
        self.ax_loss.set_title('Loss durante Training', fontsize=12, fontweight='bold')
        self.ax_loss.set_xlabel('Epoca')
        self.ax_loss.set_ylabel('Cross-Entropy Loss')
        
        # Subplot per i pesi
        self.ax_weights = self.fig.add_subplot(gs[:, 2])
        self.ax_weights.set_title('Evoluzione Pesi', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Posizioni dei neuroni
        self.neuron_positions = {
            'input': [(0, 1.5), (0, 2.5)],
            'hidden': [(1.5, 0.5), (1.5, 1.5), (1.5, 2.5), (1.5, 3.5)],
            'output': [(3, 1.5), (3, 2.5)]
        }
        
    def draw_network(self):
        """Disegna la struttura della rete neurale"""
        self.ax_network.clear()
        self.ax_network.set_xlim(-0.5, 3.5)
        self.ax_network.set_ylim(-0.5, 4.5)
        self.ax_network.set_aspect('equal')
        self.ax_network.set_title('Rete Neurale\n(2→4→2)', fontsize=14, fontweight='bold')
        self.ax_network.axis('off')
        
        # Disegna neuroni
        for layer_name, positions in self.neuron_positions.items():
            for i, (x, y) in enumerate(positions):
                if layer_name == 'input':
                    color = 'lightblue'
                    label = f'x{i+1}'
                elif layer_name == 'hidden':
                    color = 'lightgreen'
                    label = f'h{i+1}'
                else:
                    color = 'lightcoral'
                    label = f'y{i+1}'
                
                circle = Circle((x, y), 0.15, color=color, ec='black', linewidth=2)
                self.ax_network.add_patch(circle)
                self.ax_network.text(x, y, label, ha='center', va='center', fontweight='bold')
        
        # Disegna connessioni con spessore proporzionale ai pesi
        max_weight = max(np.max(np.abs(self.W1)), np.max(np.abs(self.W2)))
        
        # Connessioni input -> hidden
        for i, (x1, y1) in enumerate(self.neuron_positions['input']):
            for j, (x2, y2) in enumerate(self.neuron_positions['hidden']):
                weight = self.W1[i, j]
                width = min(abs(weight) / max_weight * 5, 5)
                color = 'red' if weight < 0 else 'blue'
                alpha = min(abs(weight) / max_weight, 1)
                self.ax_network.plot([x1+0.15, x2-0.15], [y1, y2], 
                                   color=color, linewidth=width, alpha=alpha)
        
        # Connessioni hidden -> output
        for i, (x1, y1) in enumerate(self.neuron_positions['hidden']):
            for j, (x2, y2) in enumerate(self.neuron_positions['output']):
                weight = self.W2[i, j]
                width = min(abs(weight) / max_weight * 5, 5)
                color = 'red' if weight < 0 else 'blue'
                alpha = min(abs(weight) / max_weight, 1)
                self.ax_network.plot([x1+0.15, x2-0.15], [y1, y2], 
                                   color=color, linewidth=width, alpha=alpha)
        
        # Legenda
        self.ax_network.text(0, -0.3, 'Blu: peso positivo\nRosso: peso negativo\nSpessore ∝ |peso|', 
                           fontsize=10, ha='left')
    
    def draw_data_and_decision_boundary(self):
        """Disegna i dati e il confine decisionale"""
        self.ax_data.clear()
        
        # Crea griglia per il confine decisionale
        h = 0.1
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        # Predizioni sulla griglia
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.forward_pass(grid_points)
        Z = Z[:, 1].reshape(xx.shape)  # Probabilità classe 1
        
        # Disegna il confine decisionale
        self.ax_data.contourf(xx, yy, Z, levels=50, alpha=0.3, cmap='RdBu')
        self.ax_data.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        
        # Disegna i punti
        colors = ['blue', 'red']
        for i in range(2):
            mask = self.y == i
            self.ax_data.scatter(self.X[mask, 0], self.X[mask, 1], 
                               c=colors[i], s=50, alpha=0.7, 
                               label=f'Classe {i}')
        
        # Evidenzia il campione corrente
        if hasattr(self, 'current_sample') and self.current_sample < len(self.X):
            current_x = self.X[self.current_sample]
            current_y = self.y[self.current_sample]
            self.ax_data.scatter(current_x[0], current_x[1], 
                               c='yellow', s=200, marker='*', 
                               edgecolors='black', linewidth=2,
                               label='Campione Corrente')
        
        self.ax_data.legend()
        #self.ax_data.set_xlabel('Feature 1')
        #self.ax_data.set_ylabel('Feature 2')
        #self.ax_data.set_title('Dataset e Classificazione')
        self.ax_data.grid(True, alpha=0.3)
    
    def draw_loss(self):
        """Disegna l'evoluzione della loss"""
        if len(self.training_history) > 1:
            losses = [h['loss'] for h in self.training_history]
            epochs = range(len(losses))
            
            self.ax_loss.clear()
            self.ax_loss.plot(epochs, losses, 'b-', linewidth=2)
            self.ax_loss.set_xlabel('Epoca')
            self.ax_loss.set_ylabel('Cross-Entropy Loss')
            self.ax_loss.set_title(f'Loss: {losses[-1]:.4f}')
            self.ax_loss.grid(True, alpha=0.3)
    
    def draw_weights_heatmap(self):
        """Disegna heatmap dei pesi"""
        self.ax_weights.clear()
        
        # Crea una visualizzazione separata per W1 e W2
        # W1: 2x4, W2: 4x2
        # Creiamo una matrice 6x4 per visualizzare entrambi
        max_dim = max(self.W1.shape[0] + self.W2.shape[0], 
                     max(self.W1.shape[1], self.W2.shape[1]))
        
        # Crea matrice vuota
        weights_display = np.zeros((6, 4))
        
        # Inserisci W1 (2x4) nella parte superiore
        weights_display[:self.W1.shape[0], :self.W1.shape[1]] = self.W1
        
        # Inserisci W2 (4x2) nella parte inferiore
        weights_display[2:2+self.W2.shape[0], :self.W2.shape[1]] = self.W2
        
        im = self.ax_weights.imshow(weights_display, cmap='RdBu', aspect='auto', vmin=-2, vmax=2)
        self.ax_weights.set_title('Pesi della Rete (W1: righe 0-1, W2: righe 2-5)',fontsize=10)
        self.ax_weights.set_xlabel('Indice neurone')
        self.ax_weights.set_ylabel('Connessioni')
        
        # Aggiungi i valori numerici dei pesi
        for i in range(weights_display.shape[0]):
            for j in range(weights_display.shape[1]):
                value = weights_display[i, j]
                # Mostra il valore solo se non è zero (per evitare confusione nelle zone vuote)
                if abs(value) > 1e-6:
                    # Scegli colore del testo basato sull'intensità del background
                    text_color = 'white' if abs(value) > 1.0 else 'black'
                    self.ax_weights.text(j, i, f'{value:.2f}', 
                                       ha='center', va='center', 
                                       fontsize=8, fontweight='bold',
                                       color=text_color)
        
        # Aggiungi etichette per chiarezza
        self.ax_weights.axhline(y=1.5, color='white', linewidth=2, alpha=0.7)
        self.ax_weights.text(-0.5, 0.5, 'W1', rotation=90, va='center', fontweight='bold')
        self.ax_weights.text(-0.5, 3.5, 'W2', rotation=90, va='center', fontweight='bold')
        
        # Aggiungi colorbar
        if not hasattr(self, 'colorbar'):
            self.colorbar = plt.colorbar(im, ax=self.ax_weights, shrink=0.8)
    
    def train_step(self, sample_idx):
        """Esegue un passo di training su un singolo campione"""
        x_sample = self.X[sample_idx:sample_idx+1]
        y_sample = self.y_onehot[sample_idx:sample_idx+1]
        
        # Forward pass
        y_pred = self.forward_pass(x_sample)
        
        # Calcola loss
        loss = -np.sum(y_sample * np.log(y_pred + 1e-8))
        
        # Backward pass
        self.backward_pass(x_sample, y_sample, y_pred)
        
        # Salva nella history
        self.training_history.append({
            'epoch': len(self.training_history),
            'loss': loss,
            'sample_idx': sample_idx
        })
        
        return loss
    
    def animate(self, frame):
        """Funzione di animazione"""
        # Determina quale campione stiamo processando
        sample_idx = frame % len(self.X)
        self.current_sample = sample_idx
        
        # Esegui training step
        if frame > 0:  # Salta il primo frame per mostrare stato iniziale
            loss = self.train_step(sample_idx)
            
            # Info frame
            epoch = frame // len(self.X)
            print(f"Epoca {epoch}, Campione {sample_idx}, Loss: {loss:.4f}")
        
        # Aggiorna visualizzazioni
        self.draw_network()
        self.draw_data_and_decision_boundary()
        self.draw_loss()
        self.draw_weights_heatmap()
        
        # Aggiungi info frame
        epoch = frame // len(self.X)
        self.fig.suptitle(f'Training Rete Neurale - Epoca: {epoch}, '
                         f'Campione: {sample_idx + 1}/{len(self.X)}', 
                         fontsize=10)
        
        return []
    
    def interactive_prediction(self):
        """Modalità interattiva per testare la rete addestrata"""
        print("\n" + "="*60)
        print("TRAINING COMPLETATO! Modalità Predizione Interattiva")
        print("="*60)
        print("Inserisci una coppia di numeri per testare la classificazione")
        print("La rete è stata addestrata per distinguere due classi:")
        print("- Classe 0 (blu): punti tipicamente in basso a sinistra")
        print("- Classe 1 (rosso): punti tipicamente in alto a destra")
        print("\nInserisci 'quit' per uscire")
        print("-"*60)
        
        while True:
            try:
                # Input utente
                user_input = input("\nInserisci due numeri separati da spazio (es: 1.5 -0.5): ").strip()
                
                if user_input.lower() in ['quit', 'q', 'exit']:
                    print("Arrivederci!")
                    break
                
                # Parse input
                numbers = user_input.split()
                if len(numbers) != 2:
                    print("Errore: inserisci esattamente due numeri separati da spazio")
                    continue
                
                x1, x2 = float(numbers[0]), float(numbers[1])
                
                # Normalizza l'input usando le stesse statistiche del dataset
                X_mean = self.X.mean(axis=0)
                X_std = self.X.std(axis=0)
                test_point = np.array([[x1, x2]])
                test_point_normalized = (test_point - X_mean) / X_std
                
                # Predizione
                prediction = self.forward_pass(test_point_normalized)
                prob_class_0 = prediction[0, 0]
                prob_class_1 = prediction[0, 1]
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # Output risultati
                print("\n RISULTATI PREDIZIONE:")
                print(f"   Input originale: ({x1:.2f}, {x2:.2f})")
                print(f"   Input normalizzato: ({test_point_normalized[0,0]:.2f}, {test_point_normalized[0,1]:.2f})")
                print(f"   Probabilità Classe 0 (blu): {prob_class_0:.1%}")
                print(f"   Probabilità Classe 1 (rosso): {prob_class_1:.1%}")
                print(f"   Classe predetta: {predicted_class}")
                print(f"   Confidenza: {confidence:.1%}")
                
                # Indicatore visivo della confidenza
                if confidence > 0.8:
                    print("   Predizione molto sicura!")
                elif confidence > 0.6:
                    print("   Predizione abbastanza sicura")
                else:
                    print("   Predizione incerta (vicino al confine)")
                
                # Visualizza il punto sul grafico esistente
                self.visualize_prediction(test_point_normalized[0], predicted_class, confidence)
                
            #except ValueError:
            #    print("Errore: inserisci numeri validi (es: 1.5 -0.8)")
            except KeyboardInterrupt:
                print("\nInterruzione utente. Arrivederci!")
                break
            except Exception as e:
                continue
            #    print(f"Errore: {e}")
    
    def visualize_prediction(self, point, predicted_class, confidence):
        """Visualizza il punto predetto sul grafico"""
        # Aggiorna il grafico dei dati con il nuovo punto
        self.ax_data.scatter(point[0], point[1], 
                           c='gold', s=300, marker='★', 
                           edgecolors='black', linewidth=3,
                           label=f'Predizione: Classe {predicted_class} ({confidence:.1%})',
                           zorder=10)
        
        # Aggiungi annotazione
        self.ax_data.annotate(f'Classe {predicted_class}\n{confidence:.1%}', 
                            xy=(point[0], point[1]), 
                            xytext=(10, 10), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.8),
                            fontsize=10, fontweight='bold')
        
        # Aggiorna la legenda
        self.ax_data.legend()
        
        # Refresh del plot
        plt.draw()
        plt.pause(0.1)

def main():
    # Crea il visualizzatore
    nn_viz = NeuralNetworkVisualizer()
    
    # Numero di epoche da animare
    n_epochs = 5
    total_frames = n_epochs * len(nn_viz.X)
    
    print("Avvio dell'animazione del training della rete neurale...")
    print(f"Dataset: {len(nn_viz.X)} campioni")
    print(f"Epoche: {n_epochs}")
    print(f"Frame totali: {total_frames}")
    print("Durata stimata: ~{:.0f} secondi".format(total_frames * 0.2))
    print("\nAl termine dell'animazione potrai testare la rete con i tuoi dati!")
    print("-" * 60)
    
    # Crea l'animazione
    anim = animation.FuncAnimation(
        nn_viz.fig, nn_viz.animate, frames=total_frames,
        interval=200, blit=False, repeat=False  # repeat=False per fermare alla fine
    )
    
    # Callback per quando l'animazione finisce
    def on_animation_complete(event):
        if hasattr(anim, '_step') and anim._step >= total_frames - 1:
            print("\n TRAINING COMPLETATO!")
            print(" Chiudi la finestra dell'animazione per continuare...")
    
    # Mostra l'animazione
    plt.show()
    
    # Dopo che l'animazione finisce e la finestra viene chiusa
    print("\n Avvio modalità predizione interattiva...")
    nn_viz.interactive_prediction()
    
    return anim

if __name__ == "__main__":
    # Esegui l'animazione
    anim = main()
    
    # Mantieni la finestra aperta
    plt.show()