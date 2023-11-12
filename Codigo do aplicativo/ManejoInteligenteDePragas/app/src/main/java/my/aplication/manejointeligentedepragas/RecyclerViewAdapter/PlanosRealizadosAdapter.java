package my.aplication.manejointeligentedepragas.RecyclerViewAdapter;

import android.content.Context;
//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.v7.widget.RecyclerView;
import androidx.recyclerview.widget.RecyclerView;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RelativeLayout;
import android.widget.TextView;

import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.model.PlanoAmostragemModel;

import java.util.ArrayList;
public class PlanosRealizadosAdapter extends RecyclerView.Adapter<PlanosRealizadosAdapter.ViewHolder> {

    private ArrayList<PlanoAmostragemModel> planos = new ArrayList<>();
    private Context planosContext;
    private String NomeCultura;
    private String NomeTalhao;

    public PlanosRealizadosAdapter(Context planosContext,ArrayList<PlanoAmostragemModel> planos, String NomeCultura, String NomeTalhao){
        this.planos = planos;
        this.planosContext = planosContext;
        this.NomeCultura = NomeCultura;
        this.NomeTalhao = NomeTalhao;
    }

    @NonNull
    @Override
    // resposável por inflar a view

    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.construtor_relatorio_planos_realizados, parent, false);
        ViewHolder holder = new ViewHolder(view);
        return holder;
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, final int position) {

        holder.tvAutor.setText("Autor: "+planos.get(position).getAutor());
        holder.tvData.setText("Data do plano: "+planos.get(position).getDate());
        holder.tvPlantas.setText("Número de amostras: " + planos.get(position).getPlantasAmostradas());
        holder.tvPopPragas.setText("Plantas infestadas: " + planos.get(position).getPlantasInfestadas());
        holder.tvCultura.setText("Cultura: " + NomeCultura);
        holder.tvTalhao.setText("Talhão: "+NomeTalhao);
    }

    @Override
    public int getItemCount() {
        return planos.size();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {

        TextView tvData;
        TextView tvPlantas;
        TextView tvPopPragas;
        TextView tvAutor;
        TextView tvCultura;
        TextView tvTalhao;

        RelativeLayout parent_layout_planos_realizados;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);

            tvData = itemView.findViewById(R.id.tvDataPlanosRealizados);
            tvPopPragas = itemView.findViewById(R.id.tvPopPragasPlanosRealizados);
            tvPlantas = itemView.findViewById(R.id.tvNumPlantasPlanosRealizados);
            tvAutor = itemView.findViewById(R.id.tvAutorPlano);
            parent_layout_planos_realizados = itemView.findViewById(R.id.parent_layout_planos_realizados);
            tvCultura = itemView.findViewById(R.id.tvNomeCultura);
            tvTalhao = itemView.findViewById(R.id.tvTalhaoPlano);

        }
    }

}

