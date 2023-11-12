package my.aplication.manejointeligentedepragas.RecyclerViewAdapter;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;

//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.v7.widget.RecyclerView;
import androidx.recyclerview.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import my.aplication.manejointeligentedepragas.AdicionarPropriedade;
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;
import my.aplication.manejointeligentedepragas.Cultura;
import my.aplication.manejointeligentedepragas.Propriedades;
import com.example.manejointeligentedepragas.R;
import my.aplication.manejointeligentedepragas.model.PropriedadeModel;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

public class PropriedadeCardAdapter extends RecyclerView.Adapter<PropriedadeCardAdapter.ViewHolder>{

    private static final String TAG = "PropriedadeCardAdapter";

    private ArrayList<PropriedadeModel> cards = new ArrayList<>();
    private Context propriedadeContext;
    private String nomePropriedade;

    public PropriedadeCardAdapter(Context propriedadeContext, ArrayList<PropriedadeModel> cards) {
        this.cards = cards;
        this.propriedadeContext = propriedadeContext;
    }

    @NonNull
    @Override
    // resposável por inflar a view

    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.card_propriedade, parent, false);
        ViewHolder holder = new ViewHolder(view);
        return holder;
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, final int position) {
        Log.d(TAG, "onBindViewHolder: called.");

        holder.propriedade.setText(cards.get(position).getNome());
        holder.cidade.setText("Cidade: " + cards.get(position).getCidade());
        holder.estado.setText("Estado: " + cards.get(position).getEstado());



        Controller_Usuario cu = new Controller_Usuario(propriedadeContext);
        holder.tipoUsu = cu.getUser().getTipo();



        // true para, false continua pra proxima tela. função para click demorado
        Utils u = new Utils();
        if(!u.isConected(propriedadeContext)){

        }else {
            if(holder.tipoUsu.equals("Produtor")) {
                holder.parent_layout_propriedade.setOnLongClickListener(new View.OnLongClickListener() {
                    @Override
                    public boolean onLongClick(View v) {
                        exibirCaixaDialogo(cards.get(position));
                        return true;
                    }
                });
            }

        }

        holder.parent_layout_propriedade.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(TAG, "onClick:  clicked on: ");
                Intent i = new Intent(propriedadeContext, Cultura.class);
                nomePropriedade = cards.get(position).getNome();
                // chama a intent nesse adapter
                // pega o contexr do construtor
                int codigo = cards.get(position).getCod_Propriedade();
                i.putExtra("Cod_Propriedade", codigo);
                i.putExtra("nomePropriedade", nomePropriedade);
                propriedadeContext.startActivity(i);
            }
        });


    }

    @Override
    public int getItemCount() {
        return cards.size();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {

        TextView propriedade;
        TextView estado;
        TextView cidade;

        RelativeLayout parent_layout_propriedade;

        String tipoUsu;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);

            propriedade = itemView.findViewById(R.id.tvPropriedade);
            estado = itemView.findViewById(R.id.tvEstado);
            cidade = itemView.findViewById(R.id.tvCidade);

            parent_layout_propriedade = itemView.findViewById(R.id.parent_layout_propriedade);

        }
    }

    public void exibirCaixaDialogo(final PropriedadeModel p)
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(propriedadeContext);
        dlgBox.setTitle("Alterar propriedade");
        dlgBox.setMessage("Deseja excluir ou editar a propriedade?");
        dlgBox.setPositiveButton("Editar", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                //chamar actitivity
                Intent i = new Intent(propriedadeContext, AdicionarPropriedade.class);
                i.putExtra("NomeP", p.getNome());
                i.putExtra("CidadeP", p.getCidade());
                i.putExtra("EstadoP", p.getEstado());
                i.putExtra("CodP", p.getCod_Propriedade());

                propriedadeContext.startActivity(i);
            }
        });

        dlgBox.setNegativeButton("Excluir", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                exibirCaixaConfirmacao(p);
            }
        });

        dlgBox.show();

    }

    public void exibirCaixaConfirmacao(final PropriedadeModel p)
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(propriedadeContext);
        dlgBox.setTitle("Excluindo...");
        dlgBox.setMessage("Tem certeza que deseja excluir essa propriedade?");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                ExcluirPropriedade(p.getCod_Propriedade());
            }
        });

        dlgBox.setNegativeButton("Não", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                // não faz nenhuma ação
            }
        });

        dlgBox.show();

    }

    public void ExcluirPropriedade(Integer CodP){

        String url = "https://mip.software/phpapp/excluirPropriedade.php?Cod_Propriedade="+ CodP  ;

        RequestQueue queue = Volley.newRequestQueue(propriedadeContext);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {
                //Parsing json
                //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                try {
                    JSONObject obj = new JSONObject(response);
                    boolean confirmacao = obj.getBoolean("confirmacao");
                    if(confirmacao){
                        Intent k = new Intent(propriedadeContext, Propriedades.class);
                        Toast.makeText(propriedadeContext, "Propriedade excluída com sucesso!",Toast.LENGTH_LONG).show();
                        propriedadeContext.startActivity(k);
                    }else{
                        Toast.makeText(propriedadeContext, "Propriedade não excluída! Tente novamente",Toast.LENGTH_LONG).show();
                    }

                } catch (JSONException e) {
                    Toast.makeText(propriedadeContext, e.toString(), Toast.LENGTH_LONG).show();
                }
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(propriedadeContext,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }
}
