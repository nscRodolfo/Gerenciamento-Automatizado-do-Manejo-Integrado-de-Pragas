package my.aplication.manejointeligentedepragas.RecyclerViewAdapter;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;

//import android.support.v7.widget.RecyclerView;
import androidx.recyclerview.widget.RecyclerView;
//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;

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
import my.aplication.manejointeligentedepragas.Auxiliar.Utils;
import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;
import my.aplication.manejointeligentedepragas.Cultura;
import com.example.manejointeligentedepragas.R;
import my.aplication.manejointeligentedepragas.Talhoes;
import my.aplication.manejointeligentedepragas.model.CulturaModel;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;

public class CulturaCardAdapter extends RecyclerView.Adapter<CulturaCardAdapter.ViewHolder>{

    private static final String TAG = "CulturaCardAdapter";

    private ArrayList<CulturaModel> cards = new ArrayList<>();
    private Context culturaContext;
    private int Cod_Propriedade;
    private String nomePropriedade;


    public CulturaCardAdapter(Context culturaContext, ArrayList<CulturaModel> cards, int Cod_Propriedade, String nomePropriedade) {
        this.cards = cards;
        this.culturaContext = culturaContext;
        this.Cod_Propriedade = Cod_Propriedade;
        this.nomePropriedade = nomePropriedade;
    }

    @NonNull
    @Override
    // resposável por inflar a view

    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {

        View view = LayoutInflater.from(parent.getContext()).inflate(R.layout.card_cultura, parent, false);
        ViewHolder holder = new ViewHolder(view);
        return holder;
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, final int position) {
        Log.d(TAG, "onBindViewHolder: called.");

        holder.plantaCultura.setText(cards.get(position).getnomePlanta());
        holder.numTalhao.setText("Número de talhões: " + cards.get(position).getNumeroTalhoes());
        holder.tamanhoCultura.setText("Tamanho da cultura (ha): "+cards.get(position).getTamanhoCultura());


        Controller_Usuario cu = new Controller_Usuario(culturaContext);

        holder.tipoUsu = cu.getUser().getTipo();

        Utils u = new Utils();
        if(!u.isConected(culturaContext)){

        }else {
            if (holder.tipoUsu.equals("Produtor")) {
                holder.parent_layout_cultura.setOnLongClickListener(new View.OnLongClickListener() {
                    @Override
                    public boolean onLongClick(View v) {
                        exibirCaixaConfirmacao(cards.get(position), position);
                        return true;
                    }
                });
            }
        }
        holder.parent_layout_cultura.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Log.d(TAG, "onClick:  clicked on: ");
                Intent i = new Intent(culturaContext, Talhoes.class);

                // chama a intent nesse adapter
                // pega o contexr do construtor
                i.putExtra("Cod_Cultura", cards.get(position).getCod_Cultura());
                i.putExtra("NomeCultura", cards.get(position).getnomePlanta());
                i.putExtra("Cod_Planta", cards.get(position).getFk_Cod_Planta());
                i.putExtra("Cod_Propriedade", Cod_Propriedade);
                i.putExtra("nomePropriedade", nomePropriedade);

                culturaContext.startActivity(i);
            }
        });

    }

    @Override
    public int getItemCount() {
        return cards.size();
    }

    public class ViewHolder extends RecyclerView.ViewHolder {

        TextView plantaCultura;
        TextView numTalhao;
        TextView tamanhoCultura;

        RelativeLayout parent_layout_cultura;

        String tipoUsu;

        public ViewHolder(@NonNull View itemView) {
            super(itemView);

            plantaCultura = itemView.findViewById(R.id.tvCultura);
            numTalhao = itemView.findViewById(R.id.tvNumTalhoes);
            tamanhoCultura = itemView.findViewById(R.id.tvTamanhoCultura);

            parent_layout_cultura = itemView.findViewById(R.id.parent_layout_cultura);

        }
    }



    public void exibirCaixaConfirmacao(final CulturaModel c, final Integer position)
    {
        AlertDialog.Builder dlgBox = new AlertDialog.Builder(culturaContext);
        dlgBox.setTitle("Excluindo...");
        dlgBox.setMessage("Tem certeza que deseja excluir essa cultura?");
        dlgBox.setPositiveButton("Sim", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                ExcluirCultura(c.getCod_Cultura(), position);
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

    public void ExcluirCultura(int CodC, final Integer position){

        String url = "https://mip.software/phpapp/excluirCultura.php?Cod_Cultura="+ CodC  ;

        RequestQueue queue = Volley.newRequestQueue(culturaContext);
        queue.add(new StringRequest(Request.Method.POST, url, new Response.Listener<String>() {

            @Override
            public void onResponse(String response) {
                //Parsing json
                //Toast.makeText(Entrar.this,"AQUI", Toast.LENGTH_LONG).show();
                try {
                    JSONObject obj = new JSONObject(response);
                    boolean confirmacao = obj.getBoolean("confirmacao");
                    if(confirmacao){
                        Intent k = new Intent(culturaContext, Cultura.class);
                        Toast.makeText(culturaContext, "Cultura excluída com sucesso!",Toast.LENGTH_LONG).show();
                        k.putExtra("Cod_Propriedade", cards.get(position).getFk_Cod_Propriedade());
                        k.putExtra("nomePropriedade", nomePropriedade);
                        culturaContext.startActivity(k);
                    }else{
                        Toast.makeText(culturaContext, "Cultura não excluída! Tente novamente",Toast.LENGTH_LONG).show();
                    }

                } catch (JSONException e) {
                    Toast.makeText(culturaContext, e.toString(), Toast.LENGTH_LONG).show();
                }
            }
        }, new Response.ErrorListener() {
            @Override
            public void onErrorResponse(VolleyError error) {
                Toast.makeText(culturaContext,error.toString(), Toast.LENGTH_LONG).show();
            }
        }));
    }

}
